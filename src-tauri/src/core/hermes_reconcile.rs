//! Consumer-side reconciliation for Hermes Agent profiles (#535).
//!
//! Deploying to Hermes only writes the profile's skills directory; a running
//! Hermes reader discovers that directory through its native config
//! (`HERMES_HOME/config.yaml`, `skills.external_dirs`) and caches resolutions
//! in memory. Removing the managed link therefore leaves the reader able to
//! resolve the skill until the native config stops naming it and the reader
//! reloads. This module reconciles exactly the entries the Manager owns — the
//! profile-relative references to paths it deployed — and leaves every other
//! byte of the config (native extras, comments, unrelated settings) untouched.
//!
//! The native writer (`hermes_cli.config.atomic_config_write`) is a
//! comment-preserving round-trip YAML save; the closest safe equivalent here
//! is a targeted text edit of the `skills.external_dirs` block plus an atomic
//! same-directory rename. Anything the editor cannot express with certainty
//! (flow-style lists, multi-document files, multi-line items, malformed YAML)
//! is reported as `held` and the config is left byte-identical: a held
//! reconciliation is an explicit partial result, never a silent failure and
//! never a corrupting write.

use serde::Serialize;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Component, Path, PathBuf};

/// A Hermes profile layout derived from a skills directory.
///
/// Hermes homes are `~/.hermes` (the default profile) or a directory under
/// `~/.hermes/profiles/`; the native config lives at `<home>/config.yaml` and
/// the scanned local skills dir at `<home>/skills` (upstream
/// `hermes_cli/config_home.py: get_config_path()`, `hermes_cli/profiles.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct HermesProfile {
    /// `HERMES_HOME` for this profile — `config.yaml` lives directly inside.
    pub home: PathBuf,
    /// The profile's local skills dir (`<home>/skills`).
    pub skills_dir: PathBuf,
}

impl HermesProfile {
    /// The Hermes root this installation lives under: `$HERMES_HOME` when the
    /// reader runs with it set (upstream `hermes_cli/profiles.py` honors the
    /// same variable), otherwise `~/.hermes`. Passing `--agent` a profile
    /// under a custom root therefore requires invoking the Manager with the
    /// same `HERMES_HOME`.
    fn hermes_root() -> Option<PathBuf> {
        if let Ok(home) = std::env::var("HERMES_HOME") {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
        Some(dirs::home_dir()?.join(".hermes"))
    }

    /// Recognize a Hermes profile from the skills directory a deployment
    /// targets. Only the documented layouts reconcile: the default profile
    /// (`<root>/skills`) and named profiles (`<root>/profiles/<name>/skills`).
    /// A custom agent pointed anywhere else is not claimed to be Hermes.
    pub fn from_skills_dir(skills_dir: &Path) -> Option<Self> {
        if skills_dir.file_name()? != "skills" {
            return None;
        }
        let home = skills_dir.parent()?.to_path_buf();
        let hermes_root = Self::hermes_root()?;
        let is_default = home == hermes_root;
        let is_named_profile = home.starts_with(&hermes_root.join("profiles"))
            && home
                .strip_prefix(&hermes_root)
                .is_ok_and(|rest| rest.components().count() == 2);
        if is_default || is_named_profile {
            Some(Self {
                home,
                skills_dir: skills_dir.to_path_buf(),
            })
        } else {
            None
        }
    }

    pub fn config_path(&self) -> PathBuf {
        self.home.join("config.yaml")
    }
}

/// What happened when reconciling one agent's native config.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Reconciliation {
    /// The agent key this reconciliation belongs to.
    pub agent: String,
    /// `reconciled` — config now matches the deployment (written now, or
    /// already correct); `planned` — dry-run preview of what would change;
    /// `skipped` — nothing to reconcile (no native config, no Hermes layout,
    /// no add/remove work for this agent); `held` — refused to write, config
    /// untouched, `detail` says why and what to do by hand.
    pub status: String,
    pub config_path: String,
    /// Entry strings written into `skills.external_dirs` (or that would be).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub added: Vec<String>,
    /// Entry strings removed from `skills.external_dirs` (or that would be).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<String>,
    /// A running Hermes reader keeps per-process caches; the native
    /// `agent.skill_commands.reload_skills` step is what makes the change
    /// visible to it. The Manager has no channel into that reader, so this is
    /// part of the reported contract, not something it can perform.
    pub native_reload_required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// One agent's reconciliation work for a deploy/undeploy batch.
#[derive(Debug, Clone)]
pub struct AgentReconcile {
    pub agent_key: String,
    pub profile: HermesProfile,
    /// Absolute deployment paths that must be referenced by an entry (deploy).
    pub add: Vec<PathBuf>,
    /// Absolute recorded deployment paths whose entries must be withdrawn
    /// (undeploy). Only paths this batch actually removed — a preserved path
    /// is someone else's content now, and its entry is not ours to take.
    pub remove: Vec<PathBuf>,
}

/// Reconcile one agent's native config with a batch (or preview it).
pub fn reconcile_agent(batch: &AgentReconcile, dry_run: bool) -> Reconciliation {
    let config_path = batch.profile.config_path();
    let base = Reconciliation {
        agent: batch.agent_key.clone(),
        status: String::new(),
        config_path: config_path.to_string_lossy().into_owned(),
        added: Vec::new(),
        removed: Vec::new(),
        native_reload_required: false,
        detail: None,
    };
    if batch.add.is_empty() && batch.remove.is_empty() {
        return skipped(base, "no Manager-owned references for this agent");
    }
    let raw = match fs::read_to_string(&config_path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No native config: Hermes runs on defaults, so the profile's
            // local skills dir is scanned natively and no entry is needed.
            // Creating one would put the first bytes into a file the reader
            // has never had — that decision belongs to the user.
            return skipped(
                base,
                "no native config.yaml — Hermes defaults scan the local skills dir",
            );
        }
        Err(e) => {
            return held(base, format!("config.yaml is unreadable: {e}"));
        }
    };
    match plan_entry_edits(&raw, &batch.profile, &batch.add, &batch.remove) {
        Err(reason) => held(base, reason),
        Ok(edits) if edits.added.is_empty() && edits.removed.is_empty() => Reconciliation {
            status: "reconciled".into(),
            native_reload_required: true,
            detail: Some("external_dirs already matches the deployment; no write needed".into()),
            ..base
        },
        Ok(edits) => {
            if dry_run {
                return Reconciliation {
                    status: "planned".into(),
                    added: edits.added,
                    removed: edits.removed,
                    native_reload_required: true,
                    ..base
                };
            }
            let rewritten = apply_line_edits(&raw, &edits);
            match atomic_write_preserving_mode(&config_path, &rewritten) {
                Ok(()) => Reconciliation {
                    status: "reconciled".into(),
                    added: edits.added,
                    removed: edits.removed,
                    native_reload_required: true,
                    detail: None,
                    ..base
                },
                Err(e) => held(
                    base,
                    format!(
                        "atomic write failed (config untouched): {e}; \
                         remove/add the entries by hand and reload natively"
                    ),
                ),
            }
        }
    }
}

fn skipped(mut r: Reconciliation, detail: impl Into<String>) -> Reconciliation {
    r.status = "skipped".into();
    r.detail = Some(detail.into());
    r
}

fn held(mut r: Reconciliation, detail: impl Into<String>) -> Reconciliation {
    r.status = "held".into();
    r.native_reload_required = true;
    r.detail = Some(detail.into());
    r
}

/// The line-level edits to make to one config file. Everything is decided
/// before a single byte is written.
#[derive(Debug, Default, PartialEq)]
struct EntryEdits {
    /// Lines to delete, by index into the file's line list.
    delete_lines: Vec<usize>,
    /// Content to insert, as `(original index to insert before, text)`. One
    /// entry per placement point; the text may be a multi-line block (a new
    /// key plus its items land as one unit).
    insert_lines: Vec<(usize, String)>,
    /// A line to replace wholesale (an empty `external_dirs: []` becoming a
    /// block sequence), by index.
    replace_lines: Vec<(usize, String)>,
    added: Vec<String>,
    removed: Vec<String>,
}

impl EntryEdits {
    #[cfg(test)]
    fn is_noop(&self) -> bool {
        self.delete_lines.is_empty()
            && self.insert_lines.is_empty()
            && self.replace_lines.is_empty()
    }
}

/// Decide the edits that bring `raw`'s `skills.external_dirs` in line with the
/// batch. Returns `Err` (held) for any structure this editor cannot rewrite
/// with certainty — the caller must leave the file untouched then.
fn plan_entry_edits(
    raw: &str,
    profile: &HermesProfile,
    add: &[PathBuf],
    remove: &[PathBuf],
) -> Result<EntryEdits, String> {
    // The editor reasons line-by-line; a parse failure means we cannot predict
    // what a line means, so nothing may be written.
    if let Err(e) = serde_yaml::from_str::<serde_yaml::Value>(raw) {
        return Err(format!(
            "config.yaml is not valid YAML ({e}); not touching it"
        ));
    }
    let lines: Vec<&str> = raw.lines().collect();
    if let Some(bad) = lines.iter().position(|l| {
        let t = l.trim_start();
        t == "---" || t == "..."
    }) {
        return Err(format!(
            "config.yaml is a multi-document YAML (line {}); not touching it",
            bad + 1
        ));
    }

    let skills = find_top_level_key(&lines, "skills").ok_or_else(|| {
        "no top-level 'skills' section; append one by hand or let Hermes create it".to_string()
    })?;
    if key_inline_value(lines[skills.line]).is_some_and(|inline| !inline.is_empty()) {
        return Err(format!(
            "'skills' at line {} is flow-style; not touching it",
            skills.line + 1
        ));
    }
    let section = SectionSpan::of(&lines, skills.line);
    let child_indent = section.child_indent.unwrap_or(skills.indent + 2);

    let dirs_line = section
        .key_lines
        .iter()
        .find(|(key, _)| key == "external_dirs")
        .map(|(_, line)| *line);
    let (existing_items, sequence) = match dirs_line {
        Some(line_idx) => {
            let seq = SequenceSpan::of(&lines, line_idx, child_indent)?;
            let mut items = Vec::new();
            for idx in seq.item_lines.clone() {
                let text = lines[idx].trim_start();
                let scalar = item_scalar(text.strip_prefix('-').unwrap_or(text).trim_start());
                items.push((idx, scalar));
            }
            (items, Some(seq))
        }
        None => (Vec::new(), None),
    };

    let mut edits = EntryEdits::default();

    // Removals: an entry is the Manager's to withdraw only when it expands to
    // a recorded deployment path this batch withdrew. Native extras, unrelated
    // settings and anything uninterpretable are never matched, so they are
    // never rewritten.
    for (idx, scalar) in &existing_items {
        let Some(text) = scalar else { continue };
        let expanded = expand_entry(text, profile);
        if let Some(target) = remove
            .iter()
            .find(|target| Some(normalize(target)) == expanded.as_deref().map(normalize))
        {
            edits.delete_lines.push(*idx);
            edits.removed.push(entry_spelling(target, profile));
        }
    }

    // Additions: one entry per deployment path not already referenced (by any
    // spelling that expands to the same path). Where the entry lands depends
    // on the section's shape: append to an existing block sequence, grow an
    // empty one, create the key inside the existing skills section, or create
    // a whole section at the end of the file.
    let existing_expanded: Vec<Option<PathBuf>> = existing_items
        .iter()
        .map(|(_, scalar)| {
            scalar
                .as_deref()
                .map(|s| expand_entry(s, profile))
                .flatten()
        })
        .collect();
    let missing: Vec<&PathBuf> = add
        .iter()
        .filter(|target| {
            let normalized = normalize(target);
            !existing_expanded
                .iter()
                .flatten()
                .any(|e| normalize(e) == normalized)
        })
        .collect();
    if !missing.is_empty() {
        let entry_block =
            |target: &PathBuf| -> String { format!("- {}", entry_spelling(target, profile)) };
        match sequence.as_ref() {
            Some(seq) if !seq.item_lines.is_empty() => {
                let anchor = seq
                    .last_item_line()
                    .expect("non-empty item_lines have a last item")
                    + 1;
                let block: Vec<String> = missing
                    .iter()
                    .map(|target| format!("{}{}", " ".repeat(seq.item_indent), entry_block(target)))
                    .collect();
                edits.insert_lines.push((anchor, block.join("\n")));
                edits
                    .added
                    .extend(missing.iter().map(|target| entry_spelling(target, profile)));
            }
            Some(seq) => {
                // The key exists with no active items: `external_dirs:` or an
                // empty-flow `[]` becomes a block sequence.
                let inline = key_inline_value(lines[seq.key_line]);
                if inline != Some("[]") && inline != Some("") {
                    return Err(format!(
                        "'external_dirs' at line {} is not a block or empty sequence; \
                         not touching it",
                        seq.key_line + 1
                    ));
                }
                edits.replace_lines.push((
                    seq.key_line,
                    format!("{}external_dirs:", " ".repeat(seq.key_indent)),
                ));
                let block: Vec<String> = missing
                    .iter()
                    .map(|target| {
                        format!("{}{}", " ".repeat(seq.key_indent + 2), entry_block(target))
                    })
                    .collect();
                edits
                    .insert_lines
                    .push((seq.key_line + 1, block.join("\n")));
                edits
                    .added
                    .extend(missing.iter().map(|target| entry_spelling(target, profile)));
            }
            None => {
                // The skills section has no active external_dirs key: create
                // it as the section's last child.
                let insert_at = span_insert_point(&section, lines.len());
                let mut block = vec![format!("{}external_dirs:", " ".repeat(child_indent))];
                block.extend(missing.iter().map(|target| {
                    format!("{}{}", " ".repeat(child_indent + 2), entry_block(target))
                }));
                edits.insert_lines.push((insert_at, block.join("\n")));
                edits
                    .added
                    .extend(missing.iter().map(|target| entry_spelling(target, profile)));
            }
        }
    }

    Ok(edits)
}

/// Insertion point for a new child key: just before the line that closes the
/// section, or the end of the file when the section runs to the end.
fn span_insert_point(section: &SectionSpan, end_of_lines: usize) -> usize {
    section.insert_before.unwrap_or(end_of_lines)
}

/// A top-level mapping key line: `(line index, indent)`.
struct TopKey {
    line: usize,
    indent: usize,
}

/// Find the first top-level (indent 0) mapping key with the given name,
/// ignoring comments and blanks. Only indent-0 keys qualify; nested keys of
/// the same name are not the section this editor manages.
fn find_top_level_key(lines: &[&str], name: &str) -> Option<TopKey> {
    lines.iter().enumerate().find_map(|(idx, line)| {
        if line.trim_start().is_empty() || is_comment(line) {
            return None;
        }
        let indent = leading_spaces(line);
        if indent != 0 {
            return None;
        }
        let key = line_key(line)?;
        (key == name).then_some(TopKey { line: idx, indent })
    })
}

/// One mapping section: from its key line to the next line at indent <= the
/// key's indent (comments and blanks included in the span, not terminating it).
struct SectionSpan {
    /// Where a new child key should be inserted: after the section's last
    /// content, before the line that closes it. A column-0 comment closes the
    /// section for insertion (it usually introduces the next one) even though
    /// YAML itself would not.
    insert_before: Option<usize>,
    /// Indent shared by the section's direct-child keys, when any exist.
    child_indent: Option<usize>,
    /// `(key name, line index)` for direct-child keys.
    key_lines: Vec<(String, usize)>,
}

impl SectionSpan {
    fn of(lines: &[&str], key_line: usize) -> Self {
        let indent = leading_spaces(lines[key_line]);
        let mut span = Self {
            insert_before: None,
            child_indent: None,
            key_lines: Vec::new(),
        };
        for (offset, line) in lines[key_line + 1..].iter().enumerate() {
            let idx = key_line + 1 + offset;
            let line_indent = leading_spaces(line);
            if is_comment(line) {
                if line_indent == 0 {
                    break; // introduces whatever comes next, not this section
                }
                continue;
            }
            if line.trim_start().is_empty() {
                continue;
            }
            if line_indent <= indent {
                span.insert_before = Some(idx);
                break;
            }
            if span.child_indent.is_none() {
                span.child_indent = Some(line_indent);
            }
            // Only direct children are this section's keys.
            if Some(line_indent) == span.child_indent {
                if let Some(key) = line_key(line) {
                    span.key_lines.push((key, idx));
                }
            }
        }
        span
    }
}

/// One block sequence under a mapping key.
struct SequenceSpan {
    key_line: usize,
    key_indent: usize,
    item_indent: usize,
    item_lines: Vec<usize>,
}

impl SequenceSpan {
    /// Walk the sequence under `key_line`. Any line inside it that is neither
    /// an item, a blank, a comment nor the end of the sequence (deeper content
    /// under an item — a multi-line item) is held: this editor rewrites
    /// single-line scalars only.
    fn of(lines: &[&str], key_line: usize, key_indent: usize) -> Result<Self, String> {
        let mut span = Self {
            key_line,
            key_indent,
            item_indent: 0,
            item_lines: Vec::new(),
        };
        let inline = key_inline_value(lines[key_line]);
        if let Some(inline) = inline {
            if inline == "[]" {
                return Ok(span); // nothing to scan below an empty flow sequence
            }
            if !inline.is_empty() {
                return Err(format!(
                    "'external_dirs' at line {} is flow-style; not touching it",
                    key_line + 1
                ));
            }
            // Empty inline value: a block sequence follows — scan below.
        }
        for (offset, line) in lines[key_line + 1..].iter().enumerate() {
            let idx = key_line + 1 + offset;
            if line.trim_start().is_empty() || is_comment(line) {
                continue;
            }
            let line_indent = leading_spaces(line);
            if line_indent < key_indent || (line_indent == key_indent && !is_item(line)) {
                break; // the sequence (or its section) ends here
            }
            if is_item(line) {
                if span.item_lines.is_empty() {
                    span.item_indent = line_indent;
                } else if line_indent != span.item_indent {
                    return Err(format!(
                        "inconsistent item indent at line {} in 'external_dirs'; not touching it",
                        idx + 1
                    ));
                }
                span.item_lines.push(idx);
            } else {
                // Deeper or equal-indent content under an item: a multi-line
                // item or a nested structure this editor will not guess at.
                return Err(format!(
                    "unsupported multi-line item at line {} in 'external_dirs'; not touching it",
                    idx + 1
                ));
            }
        }
        Ok(span)
    }

    fn last_item_line(&self) -> Option<usize> {
        self.item_lines.last().copied()
    }
}

fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

fn is_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("- ") || trimmed == "-"
}

fn leading_spaces(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// The mapping key of a `key: value` line (quotes stripped), if it is one.
fn line_key(line: &str) -> Option<String> {
    let colon = line.find(':')?;
    let key = line[..colon].trim();
    if key.is_empty() || key.starts_with("#-") {
        return None;
    }
    let key = key
        .strip_prefix('"')
        .and_then(|k| k.strip_suffix('"'))
        .or_else(|| key.strip_prefix('\'').and_then(|k| k.strip_suffix('\'')))
        .unwrap_or(key);
    Some(key.to_string())
}

/// Non-empty content after the `key:` colon (comments stripped from simple
/// unquoted values), used to detect flow-style values like `{...}` or `[]`.
fn key_inline_value(line: &str) -> Option<&str> {
    let colon = line.find(':')?;
    let rest = line[colon + 1..].trim();
    if rest.is_empty() || rest.starts_with('#') {
        return Some("");
    }
    // Stop at an unquoted ` #` comment.
    let value = match rest
        .starts_with('"')
        .then(|| rest)
        .or_else(|| rest.starts_with('\'').then(|| rest))
    {
        Some(quoted) => quoted,
        None => match rest.find(" #") {
            Some(hash) => rest[..hash].trim_end(),
            None => rest,
        },
    };
    Some(value)
}

/// The scalar string of a `- <scalar>` item, if it is one this editor can
/// reason about. Paths arrive quoted or plain; only a bare `~` parses as YAML
/// null, and it is still a (home) path here. Anything else — maps, booleans,
/// numbers, anchors — stays untouched by never matching.
fn item_scalar(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let (body, _rest) = split_inline_comment(text);
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    for quote in ['"', '\''] {
        if let Some(stripped) = body.strip_prefix(quote) {
            let Some(end) = stripped.find(quote) else {
                return None; // unterminated or multi-line quoting
            };
            if !stripped[end + 1..].trim().is_empty() {
                return None; // trailing content after a quoted scalar
            }
            return Some(stripped[..end].to_string());
        }
    }
    for stop in ['&', '*', '!', '{', '[', '|', '>', '%', '@', '`'] {
        if body.starts_with(stop) {
            return None;
        }
    }
    match serde_yaml::from_str::<serde_yaml::Value>(body) {
        Ok(serde_yaml::Value::String(s)) => Some(s),
        Ok(serde_yaml::Value::Null) if body == "~" => Some(body.to_string()),
        _ => None,
    }
}

fn split_inline_comment(text: &str) -> (&str, &str) {
    match text.find(" #") {
        Some(hash) => (&text[..hash], &text[hash..]),
        None => (text, ""),
    }
}

/// Expand one config entry the way the native reader does (`~` to home,
/// relative anchored at HERMES_HOME; upstream `agent/skill_utils.py:
/// _expand_path` + `_home_relative`). `${VAR}` entries are left unexpanded —
/// the Manager cannot know the reader's environment, so such an entry is
/// never matched and never rewritten.
fn expand_entry(entry: &str, profile: &HermesProfile) -> Option<PathBuf> {
    if entry.contains("${") {
        return None;
    }
    let path = Path::new(entry);
    let expanded = match (path.is_absolute(), entry.strip_prefix("~/")) {
        (true, _) => path.to_path_buf(),
        (false, Some(rest)) => dirs::home_dir()?.join(rest),
        (false, _) if entry == "~" => dirs::home_dir()?,
        (false, _) => profile.home.join(path),
    };
    Some(normalize(&expanded))
}

/// Lexical normalization (no filesystem access): collapse `.` and resolve
/// `..` without touching symlinks, so matching works even after the deployed
/// link is gone.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// How the Manager spells an entry: profile-relative when the deployment lives
/// inside the profile home (the reporter's `skills/<skill>` form), absolute
/// otherwise. Always forward-slash POSIX text, as config.yaml expects.
fn entry_spelling(target: &Path, profile: &HermesProfile) -> String {
    let normalized = normalize(target);
    let home = normalize(&profile.home);
    match normalized.strip_prefix(&home) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().replace('\\', "/"),
        _ => normalized.to_string_lossy().replace('\\', "/"),
    }
}

/// Apply decided edits to the raw text. Insert indexes name the original line
/// the new content goes immediately before (or `len` for the end); a deleted
/// line's index still stands in for "where it used to be", so an insert whose
/// target line is itself deleted lands before the next surviving line — always
/// inside the same block, never before an earlier one.
fn apply_line_edits(raw: &str, edits: &EntryEdits) -> String {
    let mut lines: Vec<String> = raw.lines().map(str::to_string).collect();
    for (idx, text) in &edits.replace_lines {
        lines[*idx] = text.clone();
    }
    let mut inserts: Vec<&(usize, String)> = edits.insert_lines.iter().collect();
    inserts.sort_by_key(|(idx, _)| *idx);
    let mut result: Vec<String> = Vec::with_capacity(lines.len() + inserts.len());
    let mut consumed = 0usize;
    for idx in 0..=lines.len() {
        while consumed < inserts.len() && inserts[consumed].0 <= idx {
            result.push(inserts[consumed].1.clone());
            consumed += 1;
        }
        if idx == lines.len() {
            break;
        }
        if edits.delete_lines.contains(&idx) {
            continue;
        }
        result.push(lines[idx].clone());
    }
    let mut out = result.join("\n");
    if raw.ends_with('\n') && !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Write via a temp file in the same directory, mode-preserved, then rename —
/// the closest local equivalent of the native `atomic_config_write` contract:
/// readers see either the old or the new file, never a partial one.
fn atomic_write_preserving_mode(path: &Path, content: &str) -> std::io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("config path has no parent directory"))?;
    let mode = fs::metadata(path)?.permissions();
    let mut tmp = tempfile::Builder::new()
        .prefix(".config.yaml")
        .suffix(".tmp")
        .tempfile_in(dir)?;
    tmp.write_all(content.as_bytes())?;
    tmp.as_file().sync_all()?;
    tmp.as_file().set_permissions(mode)?;
    tmp.persist(path).map_err(|e| e.error)?;
    // Flush the directory entry so a crash cannot resurrect the old file.
    #[cfg(unix)]
    if let Ok(dir_handle) = fs::File::open(dir) {
        let _ = dir_handle.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(home: &Path) -> HermesProfile {
        HermesProfile {
            home: home.to_path_buf(),
            skills_dir: home.join("skills"),
        }
    }

    /// A realistic fixture: the stock template's commented-out example, a
    /// native extra, an unrelated sibling setting and a Manager-owned
    /// profile-relative reference.
    const FIXTURE: &str = "\
skills:
  # Nudge the agent to create skills after complex tasks.
  creation_nudge_interval: 15

  # External skill directories — share skills across tools/agents without
  # copying them into ~/.hermes/skills/.
  external_dirs:
    # Pre-existing native extras — must survive untouched.
    - ~/.agents/skills
    - /shared/team-skills
    # Bridged by the Skills Manager deploy.
    - skills/selected-skill
  create_dir: \"\"

agent:
  max_iterations: 500
";

    fn fixture_with_entry(entry: &str) -> String {
        FIXTURE.replace("    - skills/selected-skill\n", &format!("    - {entry}\n"))
    }

    fn edits_for(
        raw: &str,
        home: &Path,
        add: &[&str],
        remove: &[&str],
    ) -> Result<EntryEdits, String> {
        plan_entry_edits(
            raw,
            &profile(home),
            &add.iter().map(|p| home.join(p)).collect::<Vec<_>>(),
            &remove.iter().map(|p| home.join(p)).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn undeploy_removes_only_the_manager_owned_entry() {
        let home = PathBuf::from("/Users/me/.hermes/profiles/work");
        let edits = edits_for(FIXTURE, &home, &[], &["skills/selected-skill"]).unwrap();
        assert_eq!(edits.removed, vec!["skills/selected-skill".to_string()]);
        assert_eq!(edits.delete_lines.len(), 1);

        let rewritten = apply_line_edits(FIXTURE, &edits);
        assert!(!rewritten.contains("skills/selected-skill"));
        // Native extras, comments and unrelated settings survive byte-for-byte.
        for kept in [
            "# External skill directories — share skills across tools/agents without",
            "# Pre-existing native extras — must survive untouched.",
            "creation_nudge_interval: 15",
            "create_dir: \"\"",
            "max_iterations: 500",
        ] {
            assert!(rewritten.contains(kept), "lost: {kept}");
        }
        // The exact diff is the one item line.
        let before: Vec<&str> = FIXTURE.lines().collect();
        let after: Vec<&str> = rewritten.lines().collect();
        let removed_lines: Vec<&str> = before
            .iter()
            .filter(|l| !after.contains(l))
            .copied()
            .collect();
        assert_eq!(removed_lines, vec!["    - skills/selected-skill"]);
        // The comment that annotated it survives (never rewrite user text we
        // did not have to); the native extras survive right below it.
        assert!(rewritten.contains("# Bridged by the Skills Manager deploy."));
        assert!(rewritten.contains("    - ~/.agents/skills\n    - /shared/team-skills"));
    }

    #[test]
    fn deploy_adds_the_profile_relative_reference_once() {
        let home = PathBuf::from("/Users/me/.hermes/profiles/work");
        let raw = FIXTURE.replace("    - skills/selected-skill\n", "");
        let edits = edits_for(&raw, &home, &["skills/selected-skill"], &[]).unwrap();
        assert_eq!(edits.added, vec!["skills/selected-skill".to_string()]);
        let rewritten = apply_line_edits(&raw, &edits);
        assert!(rewritten.contains("    - skills/selected-skill\n"));
        // Idempotent: a second reconcile is a no-op.
        let again = edits_for(&rewritten, &home, &["skills/selected-skill"], &[]).unwrap();
        assert!(again.is_noop());
    }

    #[test]
    fn entries_match_across_tilde_absolute_and_relative_spellings() {
        // `~` expands to the real user home, so the fixture profile lives
        // there too for this test.
        let home = dirs::home_dir().unwrap().join(".hermes");
        for spelling in [
            "skills/selected-skill".to_string(),
            "~/.hermes/skills/selected-skill".to_string(),
            format!("{}", home.join("skills/selected-skill").to_string_lossy()),
            "\"skills/selected-skill\"".to_string(),
        ] {
            let raw = fixture_with_entry(&spelling);
            let edits = edits_for(&raw, &home, &[], &["skills/selected-skill"]).unwrap();
            assert_eq!(edits.removed.len(), 1, "spelling {spelling} not matched");
            assert!(!apply_line_edits(&raw, &edits).contains("selected-skill\n    - /shared"));
        }
    }

    #[test]
    fn foreign_entries_are_never_touched() {
        let home = PathBuf::from("/Users/me/.hermes");
        let edits = edits_for(FIXTURE, &home, &[], &["skills/other-skill"]).unwrap();
        assert!(edits.is_noop());
        // An entry naming the same skill elsewhere (the shared library) is not
        // ours: it points at a path the Manager never deployed to.
        let raw = fixture_with_entry("/shared/library/selected-skill");
        let edits = edits_for(&raw, &home, &[], &["skills/selected-skill"]).unwrap();
        assert!(edits.is_noop());
        // Env-var entries cannot be resolved against the reader's environment.
        let raw = fixture_with_entry("${HOME}/.hermes/skills/selected-skill");
        let edits = edits_for(&raw, &home, &[], &["skills/selected-skill"]).unwrap();
        assert!(edits.is_noop());
    }

    #[test]
    fn missing_key_is_created_inside_the_existing_skills_section() {
        let home = PathBuf::from("/Users/me/.hermes");
        let raw = "\
skills:
  # Nudge the agent to create skills after complex tasks.
  creation_nudge_interval: 15
agent:
  max_iterations: 500
";
        let edits = edits_for(raw, &home, &["skills/selected-skill"], &[]).unwrap();
        assert_eq!(edits.added, vec!["skills/selected-skill".to_string()]);
        let rewritten = apply_line_edits(raw, &edits);
        assert!(rewritten.contains(
            "skills:\n  # Nudge the agent to create skills after complex tasks.\n  creation_nudge_interval: 15\n  external_dirs:\n    - skills/selected-skill\nagent:"
        ));
        assert!(serde_yaml::from_str::<serde_yaml::Value>(&rewritten).is_ok());
    }

    #[test]
    fn commented_out_example_does_not_count_as_the_key() {
        let home = PathBuf::from("/Users/me/.hermes");
        let raw = "\
skills:
  creation_nudge_interval: 15
  # external_dirs:
  #   - ~/.agents/skills
";
        let edits = edits_for(raw, &home, &["skills/selected-skill"], &[]).unwrap();
        let rewritten = apply_line_edits(raw, &edits);
        // The key is created active (as the section's last child); the
        // commented example stays exactly as it was.
        assert!(rewritten.contains("  external_dirs:\n    - skills/selected-skill\n"));
        assert!(rewritten.contains("  # external_dirs:\n  #   - ~/.agents/skills\n"));
        assert!(serde_yaml::from_str::<serde_yaml::Value>(&rewritten).is_ok());
    }

    #[test]
    fn empty_flow_sequence_grows_into_a_block_sequence() {
        let home = PathBuf::from("/Users/me/.hermes");
        let raw = "skills:\n  external_dirs: []\nagent:\n  max_iterations: 500\n";
        let edits = edits_for(raw, &home, &["skills/selected-skill"], &[]).unwrap();
        let rewritten = apply_line_edits(raw, &edits);
        assert_eq!(
            rewritten,
            "skills:\n  external_dirs:\n    - skills/selected-skill\nagent:\n  max_iterations: 500\n"
        );
    }

    #[test]
    fn flow_style_and_bad_yaml_are_held_not_rewritten() {
        let home = PathBuf::from("/Users/me/.hermes");
        // Non-empty flow sequence: refuse the whole edit.
        let edits = edits_for(
            "skills:\n  external_dirs: [~/.agents/skills]\n",
            &home,
            &[],
            &["skills/selected-skill"],
        );
        assert!(edits.is_err());
        // Malformed YAML: refuse.
        assert!(edits_for("skills: [unclosed\n", &home, &[], &["skills/x"]).is_err());
        // Multi-document YAML: refuse.
        assert!(edits_for(
            "---\nskills:\n  external_dirs:\n    - skills/x\n",
            &home,
            &[],
            &["skills/x"]
        )
        .is_err());
        // Multi-line (deeper) item under an item: refuse.
        let edits = edits_for(
            "skills:\n  external_dirs:\n    - skills/a\n      extra: 1\n",
            &home,
            &[],
            &["skills/a"],
        );
        assert!(edits.is_err());
    }

    #[test]
    fn items_at_key_indent_and_trailing_comments_are_handled() {
        let home = PathBuf::from("/Users/me/.hermes");
        let raw = "\
skills:
  external_dirs:
  - ~/.agents/skills  # team shared
  - skills/selected-skill
agent:
  max_iterations: 500
";
        let edits = edits_for(raw, &home, &[], &["skills/selected-skill"]).unwrap();
        assert_eq!(edits.removed, vec!["skills/selected-skill".to_string()]);
        let rewritten = apply_line_edits(raw, &edits);
        assert!(rewritten.contains("- ~/.agents/skills  # team shared"));
        assert!(!rewritten.contains("selected-skill"));
        assert!(serde_yaml::from_str::<serde_yaml::Value>(&rewritten)
            .unwrap()
            .get("skills")
            .is_some());
    }

    #[test]
    fn atomic_write_preserves_mode_and_replaces_in_place() {
        let tmp = tempfile::tempdir().unwrap();
        let config = tmp.path().join("config.yaml");
        fs::write(&config, "skills:\n  creation_nudge_interval: 15\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&config, fs::Permissions::from_mode(0o600)).unwrap();
            atomic_write_preserving_mode(&config, "skills:\n  external_dirs: []\n").unwrap();
            let mode = fs::metadata(&config).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        #[cfg(not(unix))]
        {
            atomic_write_preserving_mode(&config, "skills:\n  external_dirs: []\n").unwrap();
        }
        assert!(fs::read_to_string(&config)
            .unwrap()
            .contains("external_dirs"));
        // No temp leftovers.
        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".config.yaml"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn reconcile_agent_reports_each_contract_state() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let config = home.join("config.yaml");
        fs::write(&config, FIXTURE).unwrap();
        let batch = AgentReconcile {
            agent_key: "hermes".into(),
            profile: profile(&home),
            add: vec![],
            remove: vec![home.join("skills/selected-skill")],
        };

        // Dry run plans and does not write.
        let planned = reconcile_agent(&batch, true);
        assert_eq!(planned.status, "planned");
        assert_eq!(planned.removed, vec!["skills/selected-skill".to_string()]);
        assert_eq!(fs::read_to_string(&config).unwrap(), FIXTURE);

        // Real run reconciles.
        let done = reconcile_agent(&batch, false);
        assert_eq!(done.status, "reconciled");
        assert!(done.native_reload_required);
        let rewritten = fs::read_to_string(&config).unwrap();
        assert!(!rewritten.contains("selected-skill"));
        assert!(rewritten.contains("~/.agents/skills"));

        // Second run is already correct.
        let again = reconcile_agent(&batch, false);
        assert_eq!(again.status, "reconciled");
        assert!(again.added.is_empty() && again.removed.is_empty());
    }

    #[test]
    fn reconcile_agent_skips_without_config_or_without_work() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let batch = AgentReconcile {
            agent_key: "hermes".into(),
            profile: profile(&home),
            add: vec![home.join("skills/demo")],
            remove: vec![],
        };
        let no_config = reconcile_agent(&batch, false);
        assert_eq!(no_config.status, "skipped");
        assert!(
            !home.join("config.yaml").exists(),
            "no config must be created"
        );

        let empty_batch = AgentReconcile {
            agent_key: "hermes".into(),
            profile: profile(&home),
            add: vec![],
            remove: vec![],
        };
        let skipped_empty = reconcile_agent(&empty_batch, false);
        assert_eq!(skipped_empty.status, "skipped");
    }

    #[test]
    fn profile_layouts_recognize_default_and_named_profiles_only() {
        // Pin the root via Hermes's own env var so the test does not depend
        // on the machine it runs on.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join(".hermes");
        std::env::set_var("HERMES_HOME", &root);
        let hermes = root.clone();
        assert_eq!(
            HermesProfile::from_skills_dir(&hermes.join("skills")),
            Some(HermesProfile {
                home: hermes.clone(),
                skills_dir: hermes.join("skills"),
            })
        );
        let named = hermes.join("profiles").join("work");
        assert_eq!(
            HermesProfile::from_skills_dir(&named.join("skills")),
            Some(HermesProfile {
                home: named.clone(),
                skills_dir: named.join("skills"),
            })
        );
        // Not Hermes layouts.
        assert_eq!(
            HermesProfile::from_skills_dir(&tmp.path().join(".cursor/skills")),
            None
        );
        assert_eq!(
            HermesProfile::from_skills_dir(&hermes.join("skills/nested")),
            None
        );
        assert_eq!(
            HermesProfile::from_skills_dir(
                &hermes.join("profiles").join("a").join("b").join("skills")
            ),
            None
        );
        std::env::remove_var("HERMES_HOME");
    }
}

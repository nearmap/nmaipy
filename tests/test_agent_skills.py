"""The bundled Agent Skills stay loadable by the agents that read them.

Codex reads repository skills from .agents/skills, Gemini CLI from .gemini/skills or the .agents/skills alias, and
Claude Code from .claude/skills, where each skill is a symlink into .agents/skills. Every one of them reads the
SKILL.md frontmatter of the Agent Skills format (agentskills.io), so the skills keep to that format.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
SKILL_DIRS = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
# Frontmatter fields the Agent Skills format defines; tools differ in what else they accept.
SPEC_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _frontmatter(skill_md: Path) -> dict:
    """The top-level ``key: value`` lines between the opening ``---`` and the next."""
    lines = skill_md.read_text().splitlines()
    assert lines[0] == "---", f"{skill_md} should open with a '---' frontmatter line"
    end = lines.index("---", 1)
    fields = {}
    for line in lines[1:end]:
        if line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def test_skills_exist():
    assert SKILL_DIRS, f"Expected at least one skill under {SKILLS_DIR}"


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda p: p.name)
def test_frontmatter_follows_the_agent_skills_format(skill_dir):
    fields = _frontmatter(skill_dir / "SKILL.md")
    assert set(fields) <= SPEC_FIELDS, f"Fields outside the Agent Skills format: {set(fields) - SPEC_FIELDS}"
    assert fields.get("name") == skill_dir.name, "name should match the skill's directory"
    assert len(fields["name"]) <= 64 and NAME_PATTERN.fullmatch(fields["name"]), "name should be lowercase-hyphenated"
    assert 0 < len(fields.get("description", "")) <= 1024, "description should be 1-1024 characters"


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda p: p.name)
def test_claude_code_symlink_points_at_the_skill(skill_dir):
    link = REPO_ROOT / ".claude" / "skills" / skill_dir.name
    assert link.is_symlink(), f"{link} should be a symlink to {skill_dir} (on Windows, clone with symlinks enabled)"
    assert link.resolve() == skill_dir.resolve(), f"{link} should resolve to {skill_dir}"


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda p: p.name)
def test_referenced_files_exist(skill_dir):
    text = (skill_dir / "SKILL.md").read_text()
    for relative in sorted(set(re.findall(r"`(references/[\w./-]+\.md)`", text))):
        assert (skill_dir / relative).is_file(), f"SKILL.md refers to {relative}, which doesn't exist"

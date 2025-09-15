# VSCode Development Practices

## Mandatory Workspace Requirements

### **RULE: All work must be done in a workspace associated with the repo which should ALWAYS be created under c:\repos**

**Purpose**: Standardize project location and ensure proper workspace integration.

**Implementation**:
- Create all new projects under `c:\repos` directory
- Never use Desktop, Documents, or other locations
- Each project must have a `.code-workspace` file
- Workspace file must be committed to the repository

**Example**:
```bash
# Correct project setup
mkdir c:\repos\my-new-project
cd c:\repos\my-new-project
git init
# Create workspace file
echo '{"folders": [{"path": "."}]}' > my-new-project.code-workspace
```

**Violations**:
- ❌ Creating projects in Desktop
- ❌ Using Documents folder
- ❌ External workspace files not in repo

### **RULE: The project workspace should be created under the same GitHub repo as the work**

**Purpose**: Ensure workspace configuration is version controlled and shared.

**Implementation**:
- Workspace file (`.code-workspace`) lives in repository root
- Include workspace in git commits
- Configure workspace settings for project needs
- Share workspace with team members

**Workspace Template**:
```json
{
    "folders": [{"path": "."}],
    "settings": {
        "editor.tabSize": 4,
        "editor.insertSpaces": true,
        "files.trimTrailingWhitespace": true,
        "git.defaultBranchName": "main"
    },
    "extensions": {
        "recommendations": [
            "saoudrizwan.claude-dev",
            // Add project-specific extensions
        ]
    }
}
```

## VSCode Configuration Standards

### Editor Settings
```json
{
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
    },
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true
}
```

### Git Integration
```json
{
    "git.defaultBranchName": "main",
    "git.enableCommitSigning": false,
    "git.autofetch": true
}
```

### Language-Specific Settings
```json
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "python.formatting.provider": "black",
    "C_Cpp.default.cppStandard": "c++20",
    "typescript.preferences.quoteStyle": "double"
}
```

## Extension Requirements

### Mandatory Extensions
- `saoudrizwan.claude-dev` - Cline AI assistant
- `ms-vscode.vscode-json` - JSON support
- `redhat.vscode-yaml` - YAML support

### Language-Specific Extensions
- **Rust**: `rust-lang.rust-analyzer`
- **Python**: `ms-python.python`
- **C++**: `ms-vscode.cpptools`
- **PowerShell**: `ms-vscode.powershell`

### Security Extensions
- `ms-vscode.vscode-security-scan`
- `github.vscode-github-actions`

## Task Configuration

### Standard Tasks
```json
{
    "tasks": [
        {
            "label": "Build Project",
            "type": "shell",
            "command": "cargo build",
            "group": "build"
        },
        {
            "label": "Run Tests",
            "type": "shell", 
            "command": "cargo test",
            "group": "test"
        },
        {
            "label": "Security Audit",
            "type": "shell",
            "command": "cargo audit",
            "group": "test"
        }
    ]
}
```

## Debugging Configuration

### Multi-Language Debug Support
```json
{
    "configurations": [
        {
            "name": "Debug Rust",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/${workspaceFolderBasename}"
        },
        {
            "name": "Debug Python",
            "type": "python",
            "request": "launch",
            "program": "${file}"
        }
    ]
}
```

## File Organization

### Excluded Files
```json
{
    "files.exclude": {
        "**/node_modules": true,
        "**/target": true,
        "**/.git": false,
        "**/dist": true
    }
}
```

### Search Exclusions
```json
{
    "search.exclude": {
        "**/node_modules": true,
        "**/target": true,
        "**/build": true
    }
}
```

## Workspace Security

### Trust Settings
```json
{
    "security.workspace.trust.enabled": true,
    "security.workspace.trust.untrustedFiles": "prompt"
}
```

### Extension Security
- Only install extensions from verified publishers
- Review extension permissions before installation
- Regularly update extensions for security patches

## Performance Optimization

### Memory Management
```json
{
    "typescript.disableAutomaticTypeAcquisition": true,
    "search.searchOnType": false,
    "editor.minimap.enabled": false
}
```

### File Watching
```json
{
    "files.watcherExclude": {
        "**/target/**": true,
        "**/node_modules/**": true
    }
}
```

## Exception Handling

**When workspace location rule may be violated**:
- External drive development (document reason)
- Temporary testing (document cleanup plan)
- Legacy project migration (document migration timeline)

**Documentation Required**:
```markdown
# WORKSPACE-EXCEPTION: Location outside c:\repos
# REASON: [Detailed explanation]
# MITIGATION: [How this is handled]
# TIMELINE: [When this will be resolved]
```

## Compliance Checklist

- [ ] Project created under `c:\repos`
- [ ] Workspace file committed to repository
- [ ] Required extensions installed
- [ ] Standard tasks configured
- [ ] Debug configurations set up
- [ ] Security settings applied
- [ ] Performance optimizations enabled

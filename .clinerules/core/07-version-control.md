# Version Control Standards

## Branch Management

### **RULE: All GitHub work should commit to the main branch, NOT master**

**Purpose**: Use modern git conventions and avoid legacy terminology.

**Implementation**:
- Always use `main` as the primary branch name
- Configure git to default to `main` for new repositories
- Never create or use `master` branch

**Setup Commands**:
```bash
# Configure git globally to use 'main' as default branch
git config --global init.defaultBranch main

# For existing repositories, rename master to main
git branch -m master main
git push -u origin main
git push origin --delete master

# Update local references
git remote set-head origin main
```

**VSCode Configuration**:
```json
{
    "git.defaultBranchName": "main"
}
```

### **RULE: The GitHub project should always be private unless told otherwise**

**Purpose**: Protect intellectual property and maintain security by default.

**Implementation**:
- Create all new repositories as private
- Only make public when explicitly requested
- Document justification for public repositories

**GitHub Creation**:
```bash
# Using GitHub CLI (recommended)
gh repo create my-new-project --private --clone

# Or via web interface - always select "Private" option
```

**Repository Settings Checklist**:
- [ ] Repository set to Private
- [ ] Main branch protection enabled
- [ ] Require pull request reviews (for team projects)
- [ ] Require status checks to pass
- [ ] Restrict pushes to main branch

## Commit Standards

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `security`: Security improvements

**Examples**:
```bash
git commit -m "feat(api): add user authentication endpoint"
git commit -m "fix(database): resolve connection timeout issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "security(auth): implement post-quantum cryptography"
```

### Atomic Commits
- One logical change per commit
- Each commit should be buildable and testable
- Include all related files in a single commit
- Separate refactoring from feature changes

```bash
# Good: Single logical change
git add src/auth/login.rs src/auth/mod.rs tests/auth_tests.rs
git commit -m "feat(auth): implement OAuth2 login flow"

# Bad: Multiple unrelated changes
git add src/auth/ src/database/ src/ui/
git commit -m "Various updates"
```

## AI Assistant Integration

### Frequent Checkpoint Commits
- Commit at each logical milestone
- Document progress for context handoff
- Include clear descriptions of what was accomplished

```bash
# AI checkpoint commit format
git commit -m "checkpoint: implement user registration API

- Added POST /api/users endpoint
- Implemented password hashing with Argon2
- Added input validation for email and password
- Created user model and database schema
- Added unit tests for registration logic

Next: Implement email verification flow"
```

### Context Preservation
```bash
# Before reaching context limits, create detailed commit
git add .
git commit -m "checkpoint: mid-implementation of payment system

COMPLETED:
- Payment gateway integration (Stripe)
- Basic charge creation and handling
- Error handling for failed payments

IN PROGRESS:
- Webhook handling for payment events
- Database transaction management

TODO:
- Refund functionality
- Payment history API
- Frontend integration

FILES MODIFIED:
- src/payment/gateway.rs (new)
- src/payment/models.rs (updated)
- src/database/schema.rs (updated)
- tests/payment_tests.rs (new)

CONTEXT: Working on e-commerce checkout flow.
Current token usage: ~150k
Ready for task handoff if needed."
```

## Security and Compliance

### Protected Information
```bash
# Never commit these files
echo "*.key
*.pem
*.p12
*.pfx
*.crt
secrets.json
.env
.env.local
.env.production
config/secrets.*
credentials.json" >> .gitignore
```

### Signed Commits (Optional)
```bash
# Configure GPG signing (if required)
git config --global user.signingkey YOUR_GPG_KEY_ID
git config --global commit.gpgsign true
```

## Branch Protection Rules

### Main Branch Protection
```yaml
# .github/branch-protection.yml
protection_rules:
  main:
    required_status_checks:
      strict: true
      contexts:
        - "ci/tests"
        - "ci/security-scan"
        - "ci/build"
    enforce_admins: true
    required_pull_request_reviews:
      required_approving_review_count: 1
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    restrictions:
      users: []
      teams: []
```

## Release Management

### Version Tagging
```bash
# Semantic versioning tags
git tag -a v1.0.0 -m "Release version 1.0.0"
git tag -a v1.0.1 -m "Hotfix: resolve critical security issue"
git tag -a v1.1.0 -m "Feature release: add payment processing"

# Push tags to remote
git push origin --tags
```

### Release Branches (for complex projects)
```bash
# Create release branch
git checkout -b release/v1.1.0 main

# Make final adjustments
git commit -m "chore: bump version to 1.1.0"

# Merge to main
git checkout main
git merge --no-ff release/v1.1.0
git tag -a v1.1.0 -m "Release 1.1.0"

# Clean up
git branch -d release/v1.1.0
```

## Workflow Patterns

### Feature Development
```bash
# For simple features (direct to main)
git checkout main
git pull origin main
# Make changes
git add .
git commit -m "feat(feature): implement new functionality"
git push origin main
```

### Hotfix Workflow
```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/critical-security-fix
# Make fixes
git commit -m "fix(security): patch SQL injection vulnerability"
git checkout main
git merge --no-ff hotfix/critical-security-fix
git tag -a v1.0.1 -m "Security hotfix"
git push origin main --tags
git branch -d hotfix/critical-security-fix
```

## Git Configuration

### Global Settings
```bash
# Configure user information
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch to main
git config --global init.defaultBranch main

# Configure line endings (Windows)
git config --global core.autocrlf true

# Enable color output
git config --global color.ui auto

# Set default editor
git config --global core.editor "code --wait"
```

### Repository-Specific Settings
```bash
# In each repository
git config core.autocrlf true
git config pull.rebase false  # Use merge strategy
git config branch.main.remote origin
git config branch.main.merge refs/heads/main
```

## GitHub Integration

### Issue Linking
```bash
# Link commits to issues
git commit -m "fix(api): resolve timeout issue

Fixes #123
Related to #124"
```

### Pull Request Templates
```markdown
<!-- .github/pull_request_template.md -->
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Security improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Security Checklist
- [ ] No sensitive data in commits
- [ ] Dependencies updated
- [ ] Security scan passed

## AI Development Notes
- Context window usage: X%
- Checkpoint status: Complete/Partial
- Next steps documented: Yes/No
```

## Compliance Checklist

- [ ] Repository set to private
- [ ] Main branch (not master) configured
- [ ] .gitignore includes sensitive files
- [ ] Commit messages follow convention
- [ ] No secrets in commit history
- [ ] Branch protection rules enabled
- [ ] Atomic commits maintained
- [ ] Regular checkpoint commits for AI development
- [ ] Issue linking implemented
- [ ] Documentation kept current

## Exception Documentation

**When public repositories are acceptable**:
- Open source contributions
- Public documentation
- Portfolio projects
- Educational content

**Documentation Required**:
```markdown
# REPOSITORY-VISIBILITY-EXCEPTION: Public repository
# REASON: [Detailed explanation of why public is needed]
# APPROVAL: [Who approved this decision]
# REVIEW_DATE: [When to review this decision]
# RISK_MITIGATION: [How sensitive data is protected]
```

**When main branch rule may be violated**:
- Contributing to existing projects with master branch
- Legacy system constraints
- Third-party integration requirements

**Documentation Required**:
```markdown
# BRANCH-NAME-EXCEPTION: Using master branch
# REASON: [External constraint requiring master]
# TIMELINE: [When this can be resolved]
# MITIGATION: [How to minimize confusion]

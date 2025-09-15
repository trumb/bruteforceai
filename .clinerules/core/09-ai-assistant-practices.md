# AI Assistant Development Practices

## Model Configuration

### **Supported Claude Models**

Configure context window management based on the specific Claude model in use:

| Model | Identifier | Max Context | Warning | Critical | Checkpoint Frequency |
|-------|------------|-------------|---------|----------|---------------------|
| **Sonnet-4 (Primary)** | `claude-sonnet-4-20250514:1m` | 1,000k | 960k (96%) | 980k (98%) | Every 240k tokens |
| **Opus-4.1** | `claude-opus-4-1-20250805` | 200k | 150k (75%) | 180k (90%) | Every 50k tokens |
| **Sonnet-4 (Fallback)** | `claude-sonnet-4-20250514:1m` | 1,000k | 750k (75%) | 980k (98%) | Every 200k tokens |

### **Model Detection Methods**

**Option 1: Environment Variable**
```bash
# Set before starting development session
export CLAUDE_MODEL=sonnet-4    # For 1M context
export CLAUDE_MODEL=opus-4.1    # For 200k context
```

**Option 2: Session Identification**
- Check model name in Claude interface
- Note model identifier in commit messages
- Default to Opus-4.1 (conservative) if unspecified

## Context Window Management

### **RULE: Context window management must be model-dependent**

**Purpose**: Optimize development workflow based on available context window size while maintaining safety margins.

**Implementation**:
- **Sonnet-4 Primary (1M context)**: Monitor usage, take action at 960k tokens, never exceed 980k tokens
- **Sonnet-4 Fallback (1M context)**: Monitor usage, take action at 750k tokens, never exceed 980k tokens
- **Opus-4.1 (200k context)**: Monitor usage, take action at 150k tokens, never exceed 180k tokens
- Use modular file reading to minimize context consumption regardless of model

**Monitoring Commands**:
```bash
# Check current context in commit messages (model-dependent)
# For Sonnet-4 Primary (96% warning):
git commit -m "checkpoint: feature X implementation

Context: 145k/1000k tokens (14.5%)"

# For Sonnet-4 Fallback (75% warning):
git commit -m "checkpoint: feature X implementation

Context: 145k/1000k tokens (14.5%)"

# For Opus-4.1:
git commit -m "checkpoint: feature X implementation

Context: 145k/200k tokens (72.5%)"
```

**Context Reduction Strategies**:
```bash
# Read only relevant files
# Instead of reading entire codebase, target specific files
# Use search tools to find relevant code sections
# Break large files into smaller modules
```

### **RULE: At each reasonable checkpoint, commit and push the work in progress to GitHub repo with appropriate merge comments explaining what was done and why**

**Purpose**: Preserve progress remotely and enable task continuity.

**Implementation**:
- Commit and push every 50-75k tokens of context usage
- Include detailed commit messages with progress summary
- Document next steps and current status
- Tag commits for easy reference
- Always push immediately after committing to ensure remote backup

**Checkpoint Commit Format**:
```bash
git commit -m "checkpoint: [FEATURE_NAME] - [STATUS]

COMPLETED:
- [Specific accomplishment 1]
- [Specific accomplishment 2]
- [Specific accomplishment 3]

IN PROGRESS:
- [Current work item]
- [Partially completed feature]

NEXT STEPS:
- [Immediate next action]
- [Following tasks]

CONTEXT USAGE: [X]k/200k tokens ([X]%)
FILES MODIFIED: [list key files]
READY FOR: [handoff/continuation/completion]

[Additional context about decisions made or challenges encountered]"
git push origin main
```

**Example Checkpoint Commit**:
```bash
git commit -m "checkpoint: user authentication system - core complete

COMPLETED:
- User registration endpoint with email validation
- Password hashing using Argon2 with salt
- JWT token generation and validation middleware
- Database user model and migration scripts
- Unit tests for authentication flow (85% coverage)

IN PROGRESS:
- Email verification system (templates created, sending logic partial)
- Password reset functionality (database schema ready)

NEXT STEPS:
- Complete email verification implementation
- Implement password reset token generation
- Add rate limiting to auth endpoints
- Integration testing for full auth flow

CONTEXT USAGE: 147k/200k tokens (74%)
FILES MODIFIED:
- src/auth/mod.rs (new)
- src/auth/handlers.rs (new)
- src/auth/models.rs (new)
- src/middleware/jwt.rs (new)
- migrations/001_create_users.sql (new)
- tests/auth_tests.rs (new)

READY FOR: Task continuation or handoff

DECISIONS MADE:
- Chose Argon2 over bcrypt for future-proofing
- JWT stored in httpOnly cookies for security
- Email verification required before account activation"
git push origin main
```

### **RULE: Create appropriate checkpoints so that if the context window is exceeded a new task can be started to pick up where Cline left off**

**Purpose**: Enable seamless task transitions when context limits are reached.

**Implementation**:
- Create handoff documentation before reaching limits
- Use standardized checkpoint structure
- Include all necessary context for continuation
- Test resumption procedures

**Checkpoint Structure**:
```markdown
# Task Continuation Checkpoint

## Current Status
- Project: [Project Name]
- Feature: [Current Feature Being Developed]
- Progress: [X]% complete
- Context Usage: [X]k/200k tokens

## Completed Work
1. [Detailed list of completed features]
2. [Include file paths and key changes]
3. [Note any important decisions made]

## Current Work (In Progress)
- **File Being Modified**: path/to/file.rs
- **Function/Module**: specific_function_name()
- **Current State**: [describe exact state]
- **Last Working Code**: [commit hash or description]

## Immediate Next Steps
1. [Very specific next action]
2. [Following action]
3. [Third action]

## Context for New Task
- **Goal**: [What we're trying to achieve]
- **Constraints**: [Any limitations or requirements]
- **Dependencies**: [What this work depends on]
- **Testing Strategy**: [How to verify the work]

## Key Files and Locations
- Primary files: [list with brief description]
- Test files: [list test files]
- Configuration: [relevant config files]
- Documentation: [docs that need updating]

## Important Context
- [Any non-obvious design decisions]
- [Gotchas or challenges encountered]
- [External dependencies or APIs used]
- [Performance considerations]

## Verification Steps
1. [How to test current functionality]
2. [How to verify new changes]
3. [Integration test scenarios]

## Ready for New Task: YES/NO
- [ ] All work committed and pushed
- [ ] Checkpoint documentation complete
- [ ] Next steps clearly defined
- [ ] Context under 200k tokens
```

## Context Preservation Strategies

### Session Handoff Files
Create `.ai-session/` directory with:

```
.ai-session/
├── current-task.md          # Active task description
├── progress-log.md          # Detailed progress tracking
├── context-summary.md       # Key context for continuation
└── decision-log.md          # Important decisions made
```

**current-task.md Template**:
```markdown
# Current Task: [Task Name]

## Objective
[Clear, specific goal]

## Success Criteria
- [ ] [Measurable outcome 1]
- [ ] [Measurable outcome 2]
- [ ] [Measurable outcome 3]

## Current Status
- Started: [timestamp]
- Last Update: [timestamp]
- Progress: [X]%
- Context: [X]k/200k tokens

## Working Branch
- Branch: main (or specify)
- Last Commit: [commit hash]
- Files Modified: [list]

## Immediate Context
[What the AI needs to know to continue]
```

### Modular Development Approach

**File Size Limits**:
- Keep individual files under 500 lines
- Split large modules into smaller components
- Use clear module boundaries
- Minimize cross-file dependencies for context reading

**Context-Efficient Code Reading**:
```rust
// Good: Self-contained module
mod user_auth {
    // All auth-related functionality in one place
    // Clear boundaries, minimal external dependencies
}

// Bad: Scattered functionality requiring reading multiple files
```

### Strategic Checkpointing

**Model-Dependent Checkpoint Intervals**:

**For Sonnet-4 Primary (1M Context, 96% warning)**:
- **Every 240k Tokens**: Quick status commit, brief progress note
- **Every 480k Tokens**: Detailed checkpoint commit, update .ai-session/ files
- **At 720k Tokens**: Intermediate checkpoint, document progress
- **At 960k Tokens**: Major checkpoint, complete documentation update, prepare for potential handoff
- **At 970k Tokens**: Final checkpoint, consider task completion or handoff

**For Sonnet-4 Fallback (1M Context, 75% warning)**:
- **Every 200k Tokens**: Quick status commit, brief progress note
- **Every 400k Tokens**: Detailed checkpoint commit, update .ai-session/ files
- **At 750k Tokens**: Major checkpoint, complete documentation update, prepare for potential handoff
- **At 900k Tokens**: Final checkpoint, consider task completion or handoff

**For Opus-4.1 (200k Context)**:
- **Every 50k Tokens**: Quick status commit, brief progress note
- **Every 100k Tokens**: Detailed checkpoint commit, update .ai-session/ files
- **At 150k Tokens**: Major checkpoint, complete documentation update, prepare for potential handoff
- **At 175k Tokens**: Final checkpoint, consider task completion or handoff

## Task Management Patterns

### Task Decomposition
```markdown
# Large Feature: User Management System

## Subtask 1: Authentication (Target: 75k tokens)
- [ ] User registration
- [ ] Password hashing
- [ ] JWT implementation
- [ ] Basic tests

## Subtask 2: Authorization (Target: 50k tokens)
- [ ] Role-based access
- [ ] Permission system
- [ ] Middleware integration

## Subtask 3: User Profile (Target: 60k tokens)
- [ ] Profile CRUD operations
- [ ] Image upload
- [ ] Validation
```

### Progress Tracking
```bash
# Track progress in commit messages
git commit -m "progress: auth system 60% complete

Subtask 1 (Authentication): ✅ COMPLETE
Subtask 2 (Authorization): 🔄 IN PROGRESS (40%)
Subtask 3 (User Profile): ⏳ PENDING

Context: 142k/200k tokens"
```

## Code Organization for AI Development

### Self-Contained Modules
```rust
// Design modules to be understandable independently
pub mod user_management {
    //! Complete user management functionality
    //!
    //! This module contains everything needed for user operations:
    //! - Registration and authentication
    //! - Profile management
    //! - Authorization checks
    //!
    //! Dependencies: database, crypto utilities
    //! Tests: ../tests/user_management_tests.rs

    use crate::database::DbPool;
    use crate::crypto::PasswordHasher;

    // All related functionality here
}
```

### Documentation for Handoffs
```rust
/// Authentication service for user management
///
/// CONTEXT FOR AI: This service handles all user authentication.
/// - Uses Argon2 for password hashing (see crypto module)
/// - JWT tokens stored in httpOnly cookies
/// - Rate limiting applied at middleware level
///
/// CURRENT STATUS: Core functionality complete, email verification pending
/// NEXT: Implement email verification in verify_email() method
/// FILES: src/auth/mod.rs, src/auth/handlers.rs, tests/auth_tests.rs
pub struct AuthService {
    intMaxLoginAttempts: i32,
    hasher: PasswordHasher,
}
```

## Emergency Procedures

### Context Overflow Recovery
If context exceeds critical thresholds unexpectedly:

**For Sonnet-4**: If context exceeds 980k tokens unexpectedly
**For Opus-4.1**: If context exceeds 180k tokens unexpectedly

1. **Immediate Action**:
   ```bash
   git add .
   git commit -m "emergency checkpoint: context overflow at [X]k tokens

   URGENT HANDOFF NEEDED
   Current work: [brief description]
   Files modified: [list]
   Next step: [immediate next action]"
   git push origin main
   ```

2. **Create Emergency Handoff**:
   ```markdown
   # EMERGENCY HANDOFF - Context Overflow

   ## IMMEDIATE CONTEXT
   [What was being worked on when overflow occurred]

   ## LAST WORKING STATE
   - Commit: [hash]
   - Files: [list]
   - Function: [specific location]

   ## CRITICAL NEXT STEP
   [Exactly what needs to happen next]
   ```

3. **New Task Setup**:
   - Start new conversation
   - Reference emergency handoff commit
   - Resume from documented state

## Quality Assurance

### Pre-Handoff Checklist
- [ ] All changes committed and pushed to remote
- [ ] Checkpoint documentation complete
- [ ] Next steps documented
- [ ] Context usage logged
- [ ] Working state verified
- [ ] Dependencies noted
- [ ] Testing status documented

### Post-Handoff Verification
- [ ] New task can access repository
- [ ] Checkpoint documentation readable
- [ ] Next steps actionable
- [ ] Dependencies available
- [ ] Tests still passing
- [ ] No context lost

## Best Practices Summary

1. **Monitor context continuously**
2. **Commit frequently with detailed messages**
3. **Document decisions and reasoning**
4. **Prepare for handoffs proactively**
5. **Test resumption procedures**
6. **Keep modules self-contained**
7. **Use clear file organization**
8. **Maintain emergency procedures**

## Compliance Checklist

- [ ] Context window monitored
- [ ] Regular checkpoint commits
- [ ] Handoff documentation maintained
- [ ] Task decomposition applied
- [ ] Progress tracking implemented
- [ ] Emergency procedures ready
- [ ] Module boundaries clear
- [ ] Dependencies documented

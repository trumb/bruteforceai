# Code Quality Standards

## Variable Naming Requirements

### **RULE: All variables should use reverse Hungarian notation with the intended data type prepended**

**Purpose**: Provide immediate type information and improve code readability.

**Implementation**:
- Prepend data type to variable name
- Use camelCase after the type prefix
- Be consistent across the entire codebase

**Examples**:
```rust
// Rust examples
let intCounter: i32 = 0;
let strUserName: String = "Alice".to_string();
let boolIsActive: bool = true;
let vecItems: Vec<String> = vec![];
let mapUserData: HashMap<String, i32> = HashMap::new();
let floatPrice: f64 = 99.99;
let optResult: Option<String> = None;

// More complex types
let fnCallback: Box<dyn Fn(i32) -> String> = Box::new(|x| x.to_string());
let structUser: User = User::new();
let enumStatus: ConnectionStatus = ConnectionStatus::Connected;
```

```python
# Python examples
intAge: int = 25
strFirstName: str = "John"
boolIsVerified: bool = False
listItems: List[str] = []
dictConfig: Dict[str, Any] = {}
floatBalance: float = 1234.56
tupleCoordinates: Tuple[int, int] = (10, 20)

# Complex types
fnProcessor: Callable[[str], int] = lambda x: len(x)
classInstance: MyClass = MyClass()
```

```cpp
// C++ examples
int intLoopCounter = 0;
std::string strFileName = "data.txt";
bool boolHasPermission = true;
std::vector<int> vecNumbers = {1, 2, 3};
std::map<std::string, int> mapWordCount;
double floatTemperature = 98.6;
std::unique_ptr<Object> ptrInstance = std::make_unique<Object>();

// Templates and auto
auto autoResult = calculateValue(); // Use auto sparingly, prefer explicit types
template<typename T> T templateValue;
```

```typescript
// TypeScript examples  
const intUserId: number = 123;
const strApiEndpoint: string = "https://api.example.com";
const boolIsLoading: boolean = false;
const arrProducts: Product[] = [];
const objConfiguration: ConfigObject = {};
const fnAsyncHandler: (data: string) => Promise<void> = async (data) => {};

// Interface and class types
const interfaceUser: UserInterface = { id: 1, name: "Alice" };
const classManager: UserManager = new UserManager();
```

### Type Prefix Guidelines

| Type | Prefix | Example |
|------|--------|---------|
| Integer | `int` | `intCounter` |
| String | `str` | `strMessage` |
| Boolean | `bool` | `boolIsValid` |
| Float/Double | `float` | `floatPrice` |
| Array/List | `arr`/`list`/`vec` | `arrItems` |
| Object/Map | `obj`/`map`/`dict` | `objUser` |
| Function | `fn` | `fnCallback` |
| Class Instance | `class` | `classManager` |
| Interface | `interface` | `interfaceConfig` |
| Enum | `enum` | `enumStatus` |
| Pointer | `ptr` | `ptrResource` |
| Optional/Nullable | `opt` | `optValue` |

## Function Design Standards

### Function Length and Complexity
- Maximum 60 lines per function (NASA standard)
- Single responsibility principle
- Maximum 4 parameters (use structs/objects for more)
- Maximum cyclomatic complexity of 10

```rust
// Good: Single responsibility, clear naming
fn calculateTotalPrice(floatBasePrice: f64, floatTaxRate: f64) -> f64 {
    let floatTaxAmount = floatBasePrice * floatTaxRate;
    floatBasePrice + floatTaxAmount
}

// Bad: Too many responsibilities
fn processUserDataAndSendEmailAndUpdateDatabase(/* many params */) {
    // Too much happening in one function
}
```

### Error Handling
```rust
// Always use Result types in Rust
fn readConfigFile(strFilePath: &str) -> Result<Config, ConfigError> {
    let strContent = std::fs::read_to_string(strFilePath)
        .map_err(ConfigError::IoError)?;
    
    let configData = parse_config(&strContent)
        .map_err(ConfigError::ParseError)?;
    
    Ok(configData)
}
```

```python
# Use type hints and proper exception handling
def process_user_data(strUserId: str) -> Optional[UserData]:
    try:
        objUserData = fetch_user(strUserId)
        return validate_user_data(objUserData)
    except UserNotFoundError:
        logger.warning(f"User not found: {strUserId}")
        return None
    except ValidationError as e:
        logger.error(f"Validation failed for {strUserId}: {e}")
        return None
```

## Documentation Standards

### **RULE: All code should be commented as if for an experienced developer new to the language**

**Purpose**: Provide context about language-specific patterns and idioms.

**Implementation**:
- Explain language-specific patterns
- Document non-obvious design decisions
- Explain performance implications
- Reference relevant language features

```rust
// Rust-specific comments
impl User {
    /// Creates a new User instance using the builder pattern.
    /// The `?` operator propagates errors up the call stack.
    /// `Box<dyn Error>` allows returning different error types.
    pub fn create_user(strName: &str, intAge: i32) -> Result<Self, Box<dyn Error>> {
        // Validate input using Rust's match expression for exhaustive checking
        let validatedAge = match intAge {
            0..=150 => intAge,  // Range pattern matching (Rust-specific)
            _ => return Err("Invalid age range".into()),
        };
        
        // String ownership: we clone here to avoid borrowing issues
        // Alternative: accept owned String to avoid cloning
        Ok(User {
            strName: strName.to_string(),  // &str -> String conversion
            intAge: validatedAge,
        })
    }
}
```

```python
# Python-specific comments
def process_data_async(listData: List[Dict[str, Any]]) -> AsyncGenerator[ProcessedItem, None]:
    """
    Async generator function - yields items as they're processed.
    Uses list comprehension with conditional for efficient filtering.
    The 'async with' ensures proper resource cleanup.
    """
    async with aiohttp.ClientSession() as session:  # Context manager for cleanup
        # List comprehension with async iteration (Python 3.6+ feature)
        tasks = [
            process_item(session, dictItem) 
            for dictItem in listData 
            if dictItem.get('active', False)  # Dict.get() with default value
        ]
        
        # asyncio.as_completed yields tasks as they finish (not in order)
        async for task in asyncio.as_completed(tasks):
            try:
                result = await task
                yield result  # Generator yield - lazy evaluation
            except ProcessingError as e:
                # Exception chaining preserves original traceback
                raise ProcessingError(f"Failed to process item") from e
```

## Code Organization

### File Structure
```
src/
├── core/           # Core business logic
├── utils/          # Utility functions
├── types/          # Type definitions
├── errors/         # Error types
└── tests/          # Test files
```

### Import Organization
```rust
// Rust: Group imports by source
use std::collections::HashMap;  // Standard library first
use std::fs;

use serde::{Deserialize, Serialize};  // External crates
use tokio::time;

use crate::config::Config;  // Local modules last
use crate::errors::ApiError;
```

## Performance Considerations

### Memory Management
```rust
// Prefer references over cloning when possible
fn process_large_data(vecData: &Vec<LargeStruct>) -> ProcessedResult {
    // Working with references avoids copying large data structures
    vecData.iter()
        .filter(|item| item.is_valid())  // Iterator chains are zero-cost
        .map(|item| process_item(item))   // Lazy evaluation
        .collect()  // Only allocate once at the end
}
```

### Algorithmic Complexity
- Document Big O complexity for non-trivial algorithms
- Prefer O(1) lookups using HashMaps over O(n) linear searches
- Use appropriate data structures for the use case

## Testing Standards

### Test Function Naming
```rust
#[cfg(test)]
mod tests {
    // Test names should describe the scenario and expected outcome
    #[test]
    fn test_calculate_total_price_with_valid_inputs_returns_correct_sum() {
        let floatBasePrice = 100.0;
        let floatTaxRate = 0.08;
        let floatExpected = 108.0;
        
        let floatResult = calculate_total_price(floatBasePrice, floatTaxRate);
        
        assert_eq!(floatResult, floatExpected);
    }
}
```

## Exception Documentation

**When Hungarian notation may be avoided**:
- Loop counters in short scopes: `for i in 0..10`
- Well-established conventions: `self`, `ctx`, `req`, `res`
- Generic type parameters: `T`, `U`, `K`, `V`

**Documentation Required**:
```rust
// HUNGARIAN-EXCEPTION: Loop counter
// REASON: Standard convention for short-lived iterator
// SCOPE: Limited to this function only
for i in 0..intMaxIterations {
    // Process iteration
}
```

## Quality Checklist

- [ ] All variables use Hungarian notation
- [ ] Functions under 60 lines
- [ ] Single responsibility per function
- [ ] Proper error handling
- [ ] Language-specific comments provided
- [ ] Performance implications documented
- [ ] Test coverage for public functions
- [ ] Import organization follows standards

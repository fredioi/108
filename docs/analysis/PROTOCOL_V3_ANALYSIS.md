# Protocol v3.0.0 Analysis - SynapseParsingError Root Cause

## Problem Summary
SynapseParsingError occurs even with Protocol v3.0.0, despite removing compression and using simple Dict/List types.

## Complete Request/Response Flow

### 1. Dendrite (Validator) Side
```python
# dendrite.py:591-595
async with (await self.session).post(
    url=url,
    headers=synapse.to_headers(),      # Metadata + dummy objects
    json=synapse.model_dump(),         # ACTUAL FULL DATA IN HTTP BODY
    timeout=aiohttp.ClientTimeout(total=timeout),
) as response:
```

### 2. Synapse.to_headers() Processing
```python
# synapse.py:639-669
def to_headers(self) -> dict:
    headers = {"name": self.name, "timeout": str(self.timeout)}

    # ... axon/dendrite info ...

    instance_fields = self.model_dump()
    required = self.get_required_fields()  # Returns non-Optional fields

    for field, value in instance_fields.items():
        if required and field in required:
            # Create DUMMY instance for header transmission
            serialized_value = json.dumps(value.__class__.__call__())
            encoded_value = base64.b64encode(serialized_value.encode()).decode("utf-8")
            headers[f"bt_header_input_obj_{field}"] = encoded_value

    # THE PROBLEM: get_total_size() measures FULL object, not just headers!
    headers["total_size"] = str(self.get_total_size())
    return headers
```

### 3. What Goes in Headers
- `task_type` (required) → dummy string `""` → base64 `IiI=`
- `user_input` (required) → dummy string `""` → base64 `IiI=`
- **Total header size**: ~500 bytes (metadata + 2 dummy strings)

### 4. What Goes in HTTP Body
- Full `model_dump()` including:
  - `blueprint`: ~1.5KB Dict
  - `characters`: ~2KB Dict
  - `story_arc`: ~1KB Dict
  - **Total body size**: 3-5KB

### 5. Axon (Miner) Side
```python
# axon.py:717-727
body = await request.body()
request_body = body.decode()
body_dict = json.loads(request_body)  # Gets ACTUAL data from HTTP body

# Reconstruct synapse from body
syn = self.forward_class_types[request_name](**body_dict)
```

## Root Cause Identified

**The `total_size` header contains the FULL object size, not the header size!**

```python
# synapse.py:505-516
def get_total_size(self) -> int:
    self.total_size = get_size(self)  # Recursively measures ENTIRE object
    return self.total_size

# get_size() in synapse.py:19-49
def get_size(obj: Any, seen: Optional[set] = None) -> int:
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])  # Includes nested Dicts!
    # ...
```

### Error Trigger
- `total_size` header value: `"4860"` (characters task) or `"3658"` (blueprint task)
- This number represents the FULL object size (including all Dict/List data)
- The parsing code sees this large number and raises `SynapseParsingError`

## Why OCR Subnet Works

OCR subnet uses `base64_image: str` (a simple string type), not `Dict[str, Any]`:

```python
class OCRSynapse(bt.Synapse):
    base64_image: str  # Required field, 50-200KB
    response: Optional[List[dict]] = None
```

Even though `base64_image` is a required field:
1. Headers get dummy string `""` → base64 `IiI=`
2. Body gets actual 50-200KB base64 string
3. `get_total_size()` only measures a string object, not nested structures

**The key difference**: `sys.getsizeof(str)` doesn't recursively measure content, but `sys.getsizeof(dict)` + nested values does!

## Solution Options

### Option 1: Override get_total_size() (RECOMMENDED)
```python
class StoryGenerationSynapse(bt.Synapse):
    def get_total_size(self) -> int:
        """
        Override to return realistic header size, not full object size.

        Only metadata and dummy objects go in headers. Large Dict/List data
        goes in HTTP body via model_dump().
        """
        # Return small fixed value or calculate header-only size
        self.total_size = 1024  # 1KB estimate for headers
        return self.total_size
```

### Option 2: Use JSON strings instead of Dict
```python
blueprint: Optional[str] = Field(default=None)  # JSON string
characters: Optional[str] = Field(default=None)  # JSON string
```

**Downsides**:
- Loses type safety
- Requires manual JSON encoding/decoding
- Against design philosophy

### Option 3: Set total_size to 0
```python
def get_total_size(self) -> int:
    self.total_size = 0
    return 0
```

**Simplest**, but loses size monitoring capability.

## Recommended Fix

Use Option 1 with accurate header-only size calculation:

```python
import sys

class StoryGenerationSynapse(bt.Synapse):
    def get_total_size(self) -> int:
        """
        Calculate size of data transmitted in HTTP headers only.

        Bittensor sends:
        - Headers: metadata + dummy objects for required fields
        - Body: full model_dump() with actual data

        This method returns header size to prevent SynapseParsingError.
        """
        # Create copy with large fields cleared
        header_only = self.model_copy()
        header_only.blueprint = None
        header_only.characters = None
        header_only.story_arc = None
        header_only.output_data = None

        # Calculate size of remaining fields (what actually goes in headers)
        header_size = sys.getsizeof(header_only)

        # Add estimated header overhead (dendrite/axon info, etc.)
        header_size += 512

        self.total_size = header_size
        return self.total_size
```

## Implementation Status

- [x] Root cause identified
- [ ] Fix implemented
- [ ] Tested on testnet
- [ ] Verified 24-hour stability

## References

- Bittensor synapse.py:590-669 (`to_headers()`)
- Bittensor synapse.py:505-516 (`get_total_size()`)
- Bittensor dendrite.py:591-595 (HTTP POST with headers + body)
- Bittensor axon.py:717-727 (body parsing)
- OCR subnet protocol.py (successful str-based pattern)

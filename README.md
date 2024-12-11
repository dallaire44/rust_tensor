# tensor

Author: david dallaire <david@omegairisk.ca>

Note: still under construction


//! # Tensor Library

`Tensor` is a Rust library designed for multi-dimensional tensor operations using row-major memory layout.
 It provides efficient numerical computations, including broadcasting, reshaping, slicing, and element-wise
 operations, while leveraging Rust's concurrency features.

### Overview

The `Tensor` library provides:
- Multi-dimensional tensor representation using row-major order.
- Element-wise mathematical operations (e.g., addition, subtraction, division).
- Broadcasting support for operations on tensors of different shapes.
- Utility functions for slicing, reshaping, and expanding tensors.

### Features

- `Tensor`: Core data structure supporting multi-dimensional arrays.
- `RowMajorTensorShape`: Utility for managing tensor shapes and strides.
- Trait `TensorTraits`: Defines common tensor operations (e.g., logarithm, power).
- Row-Major Tensor Storage: Optimized for high performance.
- Reshaping and Expanding: Flexible manipulation of tensor shapes.
- Utilities for Cartesian product and range generation.
- Cartesian Product Utilities: Useful for generating combinations for multi-dimensional indexing.
- Concurrency Support: Utilizes `Arc<RwLock>` for shared data access.

### Example

```rust
use tensor::tensor::Tensor;

let tensor = Tensor::<f32>::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(tensor.ravel(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
tensor = "0.1.0-alpha"
```

### Modules

- `tensor`: Core tensor operations.
- `tensor_traits`: Mathematical operations on tensors.
- `numeric`: Utility functions for Cartesian products and range generation.
- `tensor_shape`: Row-major tensor shape and stride management.

### Features and API

#### `Tensor`
A struct that represents multi-dimensional tensors with the following fields:

- **data**: A shared, thread-safe vector (`Arc<RwLock<Vec<T>>>`) containing tensor elements.
- **tensor_shape**: Shape and strides of the tensor in row-major order.
- **vec_len**: Total number of elements in the tensor.

##### Example
```rust
let tensor = Tensor::<f32>::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(tensor.ravel(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

#### Operations

##### Element-Wise Subtraction
```rust
let a = Tensor::<f32>::new_raw(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
let b = Tensor::<f32>::new_raw(&[2, 2], vec![0.5, 1.0, 1.5, 2.0]);
let result = a.sub(&b);
assert_eq!(result.ravel(), vec![0.5, 1.0, 1.5, 2.0]);
```

##### Reshaping
```rust
let tensor = Tensor::<f32>::new_raw(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let reshaped = tensor.reshape(&[2, 3]);
assert_eq!(reshaped.shape(), &[2, 3]);
```

##### Broadcasting
```rust
let a = Tensor::<f32>::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::<f32>::new_raw(&[3], vec![1.0, 2.0, 3.0]);
let result = a.broadcasted_apply(&b, |a, b| a.sub(&b), false);
assert_eq!(result.ravel(), vec![0.0, 0.0, 0.0, 3.0, 3.0, 3.0]);
```
### API Overview

#### Tensor
- **`Tensor::new_raw`**: Create a new tensor from raw data.
- **`Tensor::reshape`**: Reshape the tensor to a new shape.
- **`Tensor::ravel`**: Flatten the tensor into a 1D vector.
- **`Tensor::broadcasted_apply`**: Apply binary operations with broadcasting.
- **`Tensor::expand`**: Expand tensor shape for broadcasting.
- **`Tensor::get_row`**: Retrieve a specific row from the tensor.

#### Tensor Shape Utilities
- **`RowMajorTensorShape::create_tensor_shape`**: Create tensor shape with strides.
- **`RowMajorTensorShape::reshape`**: Reshape the tensor shape.
- **`RowMajorTensorShape::expand`**: Expand the tensor shape.

#### Numerical Utilities
- **`explode_ranges_from_vector_elements`**: Generate ranges from vector elements.
- **`cartesian_product_from_shape`**: Compute the Cartesian product of ranges.

### Testing

The library is thoroughly tested. Use `cargo test` to run the test suite.

### License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).

License: MIT OR Apache-2.0

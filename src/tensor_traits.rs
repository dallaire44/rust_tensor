/// A trait defining common operations for tensor objects.
///
/// This trait provides a set of mathematical and reduction operations
/// that can be performed on tensors. Implementations of this trait are
/// expected to define how these operations are applied to the underlying
/// tensor data.
///
/// # Type Parameters
/// - `T`: The numeric type of the elements within the tensor (e.g., `f32`, `f64`).

pub trait TensorTraits<T>: {
    /// Performs element-wise subtraction between two tensors.
    ///
    /// # Arguments
    /// - `other`: A reference to another tensor of the same type and shape.
    ///
    /// # Returns
    /// A new tensor representing the result of subtracting `other` from `self`,
    /// with the same shape as the input tensors.
    ///
    /// # Panics
    /// Panics if the shapes of the tensors are incompatible for element-wise subtraction.
	fn sub(&self, other: &Self) -> Self;

    /// Performs element-wise division between two tensors.
    ///
    /// # Arguments
    /// - `other`: A reference to another tensor of the same type and shape.
    ///
    /// # Returns
    /// A new tensor representing the result of dividing `self` by `other`,
    /// with the same shape as the input tensors.
    ///
    /// # Panics
    /// Panics if the shapes of the tensors are incompatible for element-wise division
    /// or if division by zero occurs.
	fn div(&self, other: &Self) -> Self;

    /// Computes the natural logarithm (ln) of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor where each element is the natural logarithm of the corresponding
    /// element in `self`.
    ///
    /// # Panics
    /// Panics if any element of the tensor is less than or equal to zero, as the
    /// natural logarithm is undefined for these values.
    fn log(&self) -> Self;

    /// Raises each element in the tensor to the power of `n`.
    ///
    /// # Arguments
    /// - `n`: The exponent to which each element of the tensor is raised.
    ///
    /// # Returns
    /// A new tensor where each element is raised to the power of `n`.
    ///
    /// # Panics
    /// Panics if any invalid operation occurs, such as raising negative numbers to
    /// a fractional power when `T` is a floating-point type.
    fn pow(&self, n: T) -> Self;

    /// Computes the sum of elements along the specified axes of the tensor.
    ///
    /// # Arguments
    /// - `axes`: A slice of axis indices along which the summation is performed.
    ///
    /// # Returns
    /// A tensor with reduced dimensions, where the specified axes are collapsed
    /// by summing their elements.
    ///
    /// # Panics
    /// Panics if any of the axes indices are out of bounds for the shape of the tensor.
    ///
    fn sum(&self, axes: &[usize]);
}


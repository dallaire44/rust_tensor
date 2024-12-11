/// Represents the shape and strides of a tensor in row-major order.
///
/// This structure provides functionality for managing tensor shapes,
/// strides, and operations like reshaping, expansion, and data indexing
/// in row-major order. The row-major format ensures that the last dimension
/// varies the fastest in memory.
///
/// # Fields
/// - `shape`: A vector representing the dimensions of the tensor.
/// - `strides`: A vector representing the number of elements to skip in memory
///   to move to the next dimension.
/// - `offset`: The starting point in memory for the tensor data.

#[derive(Debug)]
pub struct RowMajorTensorShape {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    offset: usize,
}

impl Clone for RowMajorTensorShape {
	fn clone(&self) -> RowMajorTensorShape {
		RowMajorTensorShape {
			shape: self.shape.clone(),
			strides: self.strides.clone(),
			offset: self.offset,
		}
	}
}

impl RowMajorTensorShape {
	/// Creates a tensor shape with corresponding strides from the given dimensions.
    ///
    /// # Arguments
    /// - `shape`: A slice of dimensions representing the tensor's shape.
    ///
    /// # Returns
    /// A new `RowMajorTensorShape` instance.
    ///
    /// # Panics
    /// Panics if the provided `shape` slice is empty.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4, 5]);
    /// assert_eq!(shape.shape, vec![3, 4, 5]);
    /// assert_eq!(shape.strides, vec![20, 5, 1]);
    /// ```
	pub fn create_tensor_shape(shape: &[usize]) -> RowMajorTensorShape {
		if shape.is_empty() {
			panic!("shape is empty");
		}
		let shape = shape.to_vec();
		let mut strides: Vec<usize> = (shape[1..shape.len()]
						.iter()
						.rev())
						.into_iter()
						.scan(1, |acc, x| { *acc *= x ; Some(*acc)})
						.collect::<Vec<usize>>()
						.into_iter()
						.rev()
						.collect::<Vec<_>>();
		strides.push(1_usize);
		RowMajorTensorShape {
		    shape,
		    strides,
		    offset: 0,
		}
		      
	   } //end create_tensor_shape

	/// Retrieves the row stride (number of elements in a row) for a 2D tensor.
    ///
    /// # Returns
    /// The row stride as a `usize`.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4]);
    /// assert_eq!(shape.get_row_stride(), 4);
    /// ```   
	pub fn get_row_stride(&self) -> usize{
		let strides_len = self.strides.len();
		let mut row_stride = self.strides[0];
		if strides_len > 0 {
			row_stride = self.strides[strides_len - 2];
		}
		row_stride
	}

	/// Returns the number of dimensions in the tensor.
    ///
    /// # Returns
    /// The number of dimensions (`ndims`) as a `usize`.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4, 5]);
    /// assert_eq!(shape.ndims(), 3);
    /// ```
	pub fn ndims(&self) -> usize {
			self.shape.len()
		} 
	/// Removes dimensions of size 1 from the tensor shape.
    ///
    /// # Returns
    /// A new `RowMajorTensorShape` with all dimensions of size 1 removed.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[1, 3, 1, 5]);
    /// let reduced = shape.remove_one_dims();
    /// assert_eq!(reduced.shape, vec![3, 5]);
    /// ```
	pub fn remove_one_dims(&self) -> RowMajorTensorShape {

		let (shape, strides):(Vec<usize>, Vec<usize>) = self.shape
			.iter()
			.zip(self.strides.iter())
			.filter(|&(num, _) | *num > 1)
			.unzip();

		RowMajorTensorShape {
			shape,
			strides,
			offset: self.offset,
			}     
	}	   
	
	/// Expands the tensor shape to a new target shape.
    ///
    /// # Arguments
    /// - `shape`: The target shape to expand into.
    ///
    /// # Returns
    /// A new `RowMajorTensorShape` representing the expanded shape, or an error
    /// message if expansion is not possible.
    ///
 	pub(crate) fn expand(&self, shape: &[usize]) ->  Result<Self, String> {
       	let ndims = shape.len();
		let mut new_shape: Vec<usize> = Vec::with_capacity(ndims);
		let mut new_strides: Vec<usize> = Vec::with_capacity(ndims);
		
		for (fro_dim, to_dim) in (0..self.shape.len()).rev().zip((0..shape.len()).rev()) {
		    if self.shape[fro_dim] == shape[to_dim] {
		        new_shape.push(self.shape[fro_dim]);
		        new_strides.push(self.strides[fro_dim]);
		    } else if self.shape[fro_dim] == 1 {
		        new_shape.push(shape[to_dim]);
		        new_strides.push(0);
		    } else {
		    	
		        return Err(format!(
		            "Cannot expand tensor to shape {:?} from shape {:?}",
		            shape, self.shape
		        ));
		    }
		}
		new_shape.reverse();
		new_strides.reverse();
		
		Ok(Self {
		    shape: new_shape,
		    strides: new_strides,
		    offset: self.offset,
		})
		
	    }
	    
	/// Calculates the data index for a given set of tensor coordinates.
    ///
    /// # Arguments
    /// - `index`: A slice representing the coordinates of the tensor element.
    ///
    /// # Returns
    /// The computed data index as a `usize`.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4]);
    /// assert_eq!(shape.data_index(&[2, 3]), 11); // For 2D row-major layout
    /// ```
	pub fn data_index(&self, index: &[usize]) -> usize {
			self.offset
				+ index.iter().zip(self.strides.iter()).map(|(&i, &s)| i * s).sum::<usize>()
		}

	/// Reshapes the tensor to a new set of dimensions.
    ///
    /// # Arguments
    /// - `new_shape`: The target dimensions for reshaping.
    ///
    /// # Returns
    /// A new `RowMajorTensorShape` instance representing the reshaped tensor,
    /// or an error message if reshaping is not possible.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[6]);
    /// let reshaped = shape.reshape(&[2, 3]).unwrap();
    /// assert_eq!(reshaped.shape, vec![2, 3]);
    /// ```     
	pub fn reshape(&self, new_shape: &[usize]) -> Result<RowMajorTensorShape, String> {
		let new_len = new_shape.len();
		let mut target_strides = vec![0; new_shape.len()];
		let sparsed = self.remove_one_dims();
		let original_shape = &sparsed.shape;
		let original_strides = &sparsed.strides;
		
		let (mut original_i, mut original_j) = (0, 1);
        	let (mut new_i, mut new_j) = (0, 1);
        	
        	while (new_i < new_shape.len()) && (original_i < original_shape.len()) {
            		// First find the dimensions in both old and new we can combine -
            		// by checking that the number of elements in old and new is the same.
            		let mut np = new_shape[new_i];
            		let mut op = original_shape[original_i];
            		// This loop always ends, because we're checking that the size of old and
            		// new shapes are the same.
            		while np != op {
                		if np < op {
                    			np *= new_shape[new_j];
                    			new_j += 1;
                		} else {
                    			op *= original_shape[original_j];
                    			original_j += 1;
                		}
            		}
            		
                  	// now calculate new strides - going back to front as usual.
            		target_strides[new_j - 1] = original_strides[original_j - 1];
            		for nk in (new_i + 1..new_j).rev() {
                		target_strides[nk - 1] = target_strides[nk] * new_shape[nk];
            		}

	            	new_i = new_j;
            		new_j += 1;
            		original_i = original_j;
            		original_j += 1;	       
            	} //while loop
            	
          	let last_stride = if new_i >= 1 { target_strides[new_i - 1] } else { 1 };
        	for target_stride in target_strides.iter_mut().take(new_len).skip(new_i) {
            		*target_stride = last_stride;
        	}	
		Ok ( RowMajorTensorShape {
			shape: new_shape.to_vec(),
			strides: target_strides,
			offset: self.offset,
		})
		      				
	}

	/// Retrieves the row coordinates for a given row index.
    ///
    /// # Arguments
    /// - `index`: The row index.
    ///
    /// # Returns
    /// A 2-element array representing the start and end coordinates of the row.
    ///
    /// # Panics
    /// Panics if the index is out of bounds for the tensor.
    ///
    /// # Example
    /// ```
	/// use tensor::tensor_shape::RowMajorTensorShape;
	/// 
    /// let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4]);
    /// assert_eq!(shape.get_row_coords(1), [4, 7]);
    /// ```
	pub fn get_row_coords(&self, index: usize) -> [usize; 2] {
		let tmp = &[index].iter().zip(self.strides.iter()).map(| (&i, &s) | i * s).sum::<usize>();
		if index > self.get_row_stride() {
			panic!("get_row_coords: row coord--{}--too large.shape = {:?} -- stride = {:?}.", index, self.shape, self.strides);
		}
		
		[tmp + self.offset, tmp + self.offset + self.strides[0]-1]	
		
	}
	   
}
//end impl RowMajorTensorShape   


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tests_create_tensor_shape() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4, 5]);
		assert_eq!(shape.shape, vec![3, 4, 5]);
    	assert_eq!(shape.strides, vec![20, 5, 1]);
	}

	#[test]
	fn tests_get_row_stride() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4]);
		assert_eq!(shape.get_row_stride(), 4);
	}

	#[test]
	fn tests_ndims() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4, 5]);
		assert_eq!(shape.ndims(), 3);
	}

	#[test]
	fn tests_remove_one_dims() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[1, 3, 1, 5]);
		let reduced = shape.remove_one_dims();
		assert_eq!(reduced.shape, vec![3, 5]);
	}

	#[test]
	fn tests_expand() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[1, 3, 1]);
		let expanded = shape.expand(&[2, 3, 4]).unwrap();
		assert_eq!(expanded.shape, vec![2, 3, 4]);
	}

	#[test]
	fn tests_data_index() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4]);
		assert_eq!(shape.data_index(&[2, 3]), 11); 	
	}

	#[test]
	fn tests_reshape() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[6]);
		let reshaped = shape.reshape(&[2, 3]).unwrap();
		assert_eq!(reshaped.shape, vec![2, 3]);
	}

	#[test]
	fn tests_get_row_coords() {
		let shape = RowMajorTensorShape::create_tensor_shape(&[3, 4]);
		assert_eq!(shape.get_row_coords(1), [4, 7]);
	}

}

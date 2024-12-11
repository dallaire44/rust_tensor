#![allow(non_camel_case_types)]
#![allow(unused_imports)]
use std::sync::{Arc, RwLock};
use std::ops::{Div, Sub, Mul, Add, FnMut, Range, RangeInclusive};

use std::iter;
use crate::tensor_shape::RowMajorTensorShape;
use crate::tensor_traits::TensorTraits;
use crate::numeric::{ cartesian_product, cartesian_product_from_shape };
use num_traits::{Float};
//use std::ops:Neg;


//RwLock allows multiple readers to access the data concurrently without requiring a full lock. 
//It provides two types of locks: read locks and write locks. Multiple read locks can be 
//acquired at the same time, but a write lock prevents both read and write operations during its lifetime.

//Shared Read Access: An RwLock allows multiple threads to concurrently acquire read locks, providing 
//shared read access to the protected data. This is useful when you have many threads that need to read the data simultaneously.


/// A multi-dimensional tensor with support for mathematical operations and broadcasting.
///
/// The `Tensor` struct represents a tensor with its data stored in a row-major order.
/// It supports element-wise operations, reshaping, slicing, broadcasting, and more.
/// The underlying data is managed with an `Arc<RwLock<Vec<T>>>` to allow concurrent
/// reads and exclusive writes.
///
/// # Type Parameters
/// - `T`: The numeric type of the elements within the tensor (e.g., `f32`, `f64`).
///
/// # Fields
/// - `data`: Shared data storage for the tensor, allowing concurrent read and exclusive write access.
/// - `tensor_shape`: Encapsulates the shape and strides of the tensor.
/// - `vec_len`: The total number of elements in the tensor.
///
/// # Examples
/// ```
/// use tensor::tensor::Tensor;
/// 
/// let tensor = Tensor::<f32>::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(tensor.shape(), &[2, 3]);
/// assert_eq!(tensor.ravel(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
#[derive(Debug)]
pub struct Tensor<T> {
    pub data: Arc<RwLock<Vec<T>>>,             //row major, allow mut and multi thread 
    pub tensor_shape: RowMajorTensorShape,       // higher/outer dimension is on the lower index.
    pub vec_len: usize,
}

//impl<'a: a + Copy + std::fmt::Debug + std::ops::Sub<Output = T> + std::iter::Sum<&'a T> > Tensor<T> 
//impl<'a, T> Tensor<T>
impl<'a, T> Tensor<T>
where
    //T: 'a + Copy + std::fmt::Debug + std::ops::Sub<Output = T> + std::iter::Sum<&'a T>,
    //T: 'a + Copy + std::fmt::Debug + std::ops::Div<Output = T> + std::ops::Sub<Output = T> + std::iter::Sum<&'a T>,
    //T: 'a + Copy + std::fmt::Debug + std::ops::Div<Output = T> + std::ops::Sub<Output = T> + std::iter::Sum<&'a T> + Float,
    T: 'a + Copy + std::fmt::Debug + std::ops::Div<Output = T> + std::ops::Sub<Output = T>,
    Vec<T>: FromIterator<T>,
    T: Float,
    {

	/// Creates a new tensor from the given shape and data.
    ///
    /// # Arguments
    /// - `shape`: A slice of dimensions representing the tensor's shape.
    /// - `data`: A vector containing the tensor's data in row-major order.
    ///
    /// # Returns
    /// A new instance of `Tensor`.
    ///
    /// # Panics
    /// Panics if the provided data length does not match the total number of elements
    /// implied by the shape.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
	pub fn new_raw(shape: &[usize], data: Vec<T>) -> Self {
        let new_data = Arc::new(RwLock::new(data.to_vec()));
        let tensor_shape = RowMajorTensorShape::create_tensor_shape(shape);
        let data_vec_len = data.len();
        Tensor {
            data: new_data,
            tensor_shape: tensor_shape,
            vec_len: data_vec_len,
        }
    }
    
    /// Creates a new tensor with the same shape as the current tensor
    /// but replaces its data with the provided vector.
    ///
    /// # Arguments
    /// - `data`: A vector containing the new data in row-major order.
    ///
    /// # Returns
    /// A new `Tensor` with the same shape but the provided data.
    ///
    fn with_contiguous_data(&self, data: Vec<T>) -> Self {
    //fn with_contiguous_data(&self, data: &[T]) -> Self {
        Self::new_raw(self.shape(), data)
    }

    /// Flattens the tensor into a contiguous vector.
    ///
    /// # Returns
    /// A `Vec<T>` containing all the elements of the tensor in row-major order.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(tensor.ravel(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```   
    pub fn ravel(&self) -> Vec<T> {
		let read_only_access = self.data.read().unwrap();
		read_only_access.to_vec()
    }

    /// Retrieves the shape of the tensor.
    ///
    /// # Returns
    /// A slice representing the dimensions of the tensor.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.tensor_shape.shape
    }

    /// Retrieves the strides of the tensor.
    ///
    /// # Returns
    /// A slice representing the strides of the tensor in row-major order.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(tensor.strides(), &[3, 1]);
    /// ```
    pub fn strides(&self) -> &[usize] {
        &self.tensor_shape.strides
    }   

    /// Creates a new tensor with the same data but a different shape.
    ///
    /// # Arguments
    /// - `tensor_shape`: A `RowMajorTensorShape` representing the new shape.
    ///
    /// # Returns
    /// A new `Tensor` with the same data but the new shape.
    fn with_tensor_shape(&self, tensor_shape: RowMajorTensorShape) -> Self {
        Self {
            data: Arc::clone(&self.data),
            tensor_shape,
            vec_len: self.vec_len,
        }
    }   

    pub fn zip(&self, other: &Self, closure: impl Fn(T, T) -> T) -> Self {

        let target_shape = &self.tensor_shape.shape;
 
        let cart = cartesian_product_from_shape(target_shape);
                
        let lock_self_data = self.data.read().unwrap();
        let lock_other_data = other.data.read().unwrap();

 
        let result: Vec<_> = cart.iter().map(| shape | (self.tensor_shape.data_index(&shape), 
                                                other.tensor_shape.data_index(&shape))).
                                                map(| (x,y) | (lock_self_data[x], lock_other_data[y])).
                                                map(| (x, y) | closure(x, y)).collect();

        let ntensor_shape = RowMajorTensorShape::create_tensor_shape(target_shape);

        let ret = Self {
                data: Arc::new(RwLock::new(result)),
                tensor_shape: ntensor_shape,
                vec_len: self.vec_len,
            };	
        ret		
            
    }

    /// Expands the tensor's shape to a new target shape.
    ///
    /// # Arguments
    /// - `shape`: The target shape for expansion.
    ///
    /// # Returns
    /// A new tensor with the expanded shape.
    ///
    /// # Panics
    /// Panics if the target shape is incompatible with the current tensor's shape.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[1, 3], vec![1.0, 2.0, 3.0]);
    /// let expanded = tensor.expand(&[2, 3]);
    /// assert_eq!(expanded.shape(), &[2, 3]);
    /// ```
    pub fn expand(&self, shape: &[usize]) ->  Self {
	    let _tensor_shape = self.tensor_shape.expand(shape).unwrap();
	    self.with_tensor_shape(_tensor_shape)
	 }    

     /// Reshapes the tensor to a new set of dimensions.
    ///
    /// # Arguments
    /// - `new_shape`: The target dimensions for reshaping.
    ///
    /// # Returns
    /// A new `Tensor` instance with the specified shape.
    ///
    /// # Panics
    /// Panics if the total number of elements in the new shape does not match
    /// the original number of elements.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let reshaped = tensor.reshape(&[2, 3]);
    /// assert_eq!(reshaped.shape(), &[2, 3]);
    /// ```
     pub fn reshape(&self, _new_shape: &[usize]) -> Self{
		if _new_shape.iter().product::<usize>() != self.tensor_shape.shape.iter().product::<usize>() {
			panic!("reshape expects the same number of elements {:?}, {:?}", self.tensor_shape.shape, _new_shape);
		}
		let _tensor_shape = self.tensor_shape.reshape(_new_shape).unwrap();
		self.with_tensor_shape(_tensor_shape)
	} 


	pub fn reshape2(&self, _new_shape: &[usize]) -> Self{	
		if _new_shape.iter().product::<usize>() != self.tensor_shape.shape.iter().product::<usize>() {
			panic!("reshape expects the same number of elements {:?}, {:?}", self.tensor_shape.shape, _new_shape);
		}
		let _tensor_shape = RowMajorTensorShape::create_tensor_shape(_new_shape);

		self.with_tensor_shape(_tensor_shape)
	} 

    /// Performs element-wise subtraction with another tensor.
    ///
    /// # Arguments
    /// - `other`: A reference to another tensor of the same type.
    ///
    /// # Returns
    /// A new tensor representing the element-wise difference.
    ///
    /// # Panics
    /// Panics if the shapes of the tensors are incompatible for element-wise subtraction.
    ///
    fn sub(&self, other: &Self) -> Self {
        self.zip(other, |x, y| x - y)
    }	
    
    /// Performs element-wise division with another tensor.
    ///
    /// # Arguments
    /// - `other`: A reference to another tensor of the same type.
    ///
    /// # Returns
    /// A new tensor representing the element-wise quotient.
    ///
    /// # Panics
    /// Panics if the shapes of the tensors are incompatible for element-wise division
    /// or if division by zero occurs.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// use crate::tensor::tensor_traits::TensorTraits;
    /// 
    /// let a = Tensor::new_raw(&[2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// let b = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = a.div(&b);
    /// assert_eq!(result.ravel(), vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    /// ```
    fn div(&self, other: &Self) -> Self {
        self.zip(other, |x, y| x / y)
    }	

    /// Applies a binary operation to two tensors with broadcasting.
    ///
    /// This method supports broadcasting, where tensors with different shapes
    /// can still perform operations if their dimensions are compatible.
    ///
    /// # Arguments
    /// - `other`: A reference to the second tensor.
    /// - `f`: A closure that defines the binary operation.
    /// - `reverse`: A boolean indicating whether to reverse the operation.
    ///
    /// # Returns
    /// A new tensor with the result of the broadcasted operation.
    ///
    fn broadcasted_apply(
        &self,
        other: &Self,

        f: impl Fn(&Self, &Self) -> Self,

        reverse: bool,
    ) -> Self {   
        
        if self.tensor_shape.ndims() > other.tensor_shape.ndims() {
            // Rust tidbit: I originally did not have a reverse parameter,
            // but just called |a,b| f(b,a) in the recursive call. This doesn't work,
            // because it hits the recursion limit: https://stackoverflow.com/questions/54613966/error-reached-the-recursion-limit-while-instantiating-funcclosure
            return other.broadcasted_apply(self, f, !reverse);
        }
        
        if self.tensor_shape.ndims() == other.tensor_shape.ndims() {
            let res_shape = self
                .tensor_shape
                .shape
                .iter()
                .zip(other.tensor_shape.shape.iter())
                .map(|(a, b)| *a.max(b))
                .collect::<Vec<_>>();
		
            let s_expanded = self.expand(&res_shape);
            let o_expanded = other.expand(&res_shape);
            if reverse {
				let wtf = f(&o_expanded, &s_expanded);
                return wtf;
            }
            return f(&s_expanded, &o_expanded);
            
        } 
		
        let num_ones_to_add = other.tensor_shape.shape.len().saturating_sub(self.tensor_shape.shape.len());
        let mut new_shape = vec![1; num_ones_to_add];
        
        new_shape.extend(&self.tensor_shape.shape);

        self.reshape(&new_shape)
            .broadcasted_apply(other, f, reverse)
        
    }	

    pub fn print_row(&self, index: usize){
        if index > self.tensor_shape.get_row_stride() {
			panic!("print_row: row coord--{}--too large.shape = {:?} -- stride = {:?}.", index, self.shape(), self.strides());
		}
		println!("row coord: {:?}",self.tensor_shape.get_row_coords(index));
		println!("row coord2: {:?}",self.get_row(index));
	}	
    
    pub fn get_value(&self, n: usize) -> T {
        let lock_self_data = self.data.read().unwrap();
        lock_self_data[n]
    }

    /// Retrieves a specific row from a 2D tensor.
    ///
    /// # Arguments
    /// - `n`: The index of the row to retrieve.
    ///
    /// # Returns
    /// A vector containing the elements of the specified row.
    ///
    /// # Panics
    /// Panics if the tensor is not 2D or if the index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use tensor::tensor::Tensor;
    /// 
    /// let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(tensor.get_row(1), vec![4.0, 5.0, 6.0]);
    /// ```
	pub fn get_row(&self, n: usize) -> Vec<T>  {
        if n > self.tensor_shape.get_row_stride() {
			panic!("get_row: row coord--{}--too large.shape = {:?} -- stride = {:?}.", n, self.shape(), self.strides());
		}
		let coords = self.tensor_shape.get_row_coords(n);
		
		let lock_self_data = self.data.read().unwrap();
		let part: Vec<T> = lock_self_data[coords[0]..=coords[1]].iter().cloned().collect();
		//https://users.rust-lang.org/t/efficient-way-of-copying-partial-content-from-vector/8317/8
		part
	}

	pub fn set_row(&self, n: usize, data: &[T]) {
        if n > self.tensor_shape.get_row_stride() {
			panic!("set_row: row coord--{}--too large.shape = {:?} -- stride = {:?}.", n, self.shape(), self.strides());
		}
		let coords = self.tensor_shape.get_row_coords(n);
		let mut lock_data = self.data.write().unwrap();
		lock_data[coords[0]..=coords[1]].copy_from_slice(&data);
    }   
    
    pub fn _elementwise<F>(&self, closure: F) -> Tensor<T>
    where 
        F: Fn(&T) -> T,
    {   
        let lock_data = self.data.write().unwrap();

        let mut new_data = Vec::new();
        for i in &*lock_data {
            new_data.push(closure(i));
        }
        self.with_contiguous_data(new_data)
        
    } 

    pub fn _elementwise2<F>(&self, closure: F, n: T) -> Tensor<T>
    where 
        F: Fn(&T, T) -> T,
    {
        let lock_data = self.data.write().unwrap();

        let mut new_data = Vec::new();
        for i in &*lock_data {
            new_data.push(closure(i, n));
        }
        self.with_contiguous_data(new_data)

    }

    pub fn slice2(&self, input_vectors: &[Vec<i32>]) -> Vec<T> {
        //let _vec_vec = slice.iter().map(| i | i.clone().collect::<Vec<_>>()).collect::<Vec<_>>();
        let _cart_prod = cartesian_product(&input_vectors);
        let _uarray: Vec<_> = _cart_prod.iter().map(| inner | inner.iter().map(| &x | x as usize).collect::<Vec<_>>()).collect();
        _uarray.iter().map(| x | self.tensor_shape.data_index(&x)).map( | j | self.get_value(j)).collect()
    }

    pub fn slice(&self, slice: &[Range<usize>]) -> Vec<T> {
		println!("slicing");
        
        //takes slice = [0..1,0..1,0..3] and creates _vec_vec [[0], [0], [0, 1, 2]]
        let _vec_vec = slice.iter().map(| i | i.clone().collect::<Vec<_>>()).collect::<Vec<_>>();
        //self.slice2(&_vec_vec)
        let _cart_prod = cartesian_product(&_vec_vec);
        let _uarray: Vec<_> = _cart_prod.iter().map(| inner | inner.iter().map(| &x | x as usize).collect::<Vec<_>>()).collect();
        _uarray.iter().map(| x | self.tensor_shape.data_index(&x)).map( | j | self.get_value(j)).collect()
        }

     
    fn collapse(&self, _f: impl Fn(T, T) -> T, axes: &[usize]) {

        println!("collapse {:?}", axes);
        println!("shape {:?}", self.shape());
        let mut _ranges: Vec<_> = self.shape().to_vec();
        _ranges.remove(axes[0]);
        let mut _new_ranges: Vec<_> = _ranges.iter().map(| x | (0..*x).collect::<Vec<_>>()).collect::<Vec<_>>();
        //_new_ranges[axes[0]] = vec![_ranges[axes[0]]];
        //_ranges.push(vec![self.shape().to_vec()[axes[0]]]);
        //input_vector.iter().map(| &i | (0..i).collect::<Vec<_>>()).collect::<Vec<_>>()
        println!("new_ranges {:?}", _new_ranges);

        //let _target_range_vec: Vec<_> = (0..self.shape()[axes[0]]).collect::<Vec<_>>();//.iter().map(| x | *x as usize).collect();
        let _target_range = 0..self.shape()[axes[0]];
        println!("_target_range_vec {:?}", _target_range);


        //let _target_range_vec_ref: Vec<&usize> = _target_range_vec.iter().collect();


        let _outer_carts = cartesian_product(&_new_ranges);
        println!("_outer_carts {:?}", _outer_carts);
        _outer_carts.iter().for_each(| x | {  
                                                            println!("x {:?}", x);
                                                            let mut t: Vec<_> = x.iter().map(| j | *j..(*j+1)).collect();
                                                            //let mut t: Vec<_> = x.iter().map(|j| vec![j]).collect();
                                                            t.insert(axes[0], _target_range.clone());
                                                            println!("t {:?} ",t);
                                                            println!("self.slice {:?}", self.slice(t.as_slice()));
                                                            //let _sub_vec = self.slice(t.as_slice()).clone();
                                                            //println!("sum : {:?}", _sub_vec.iter().sum::<T>());


                                                        });
 
        

        //println!("_itit {:?}", _itit);
        //let _huh = _outer_carts.iter().map()


    }
    /* Chat GPT suggestion with sum implemented */
    /* 
    pub fn collapse(&self, f: impl Fn(T, T) -> T, axes: &[usize]) -> Tensor<T> {
        let mut result_shape = self.tensor_shape.shape.clone();
        for &axis in axes {
            result_shape[axis] = 1;
        }

        let mut new_data = Vec::new();
        let cartesian = cartesian_product_from_shape(&result_shape);

        for coords in cartesian {
            let mut slice_coords: Vec<_> = self.tensor_shape.shape.iter().map(|&dim| 0..dim).collect();
            for (i, &coord) in coords.iter().enumerate() {
                slice_coords[i] = coord..coord + 1;
            }

            let slice = self.slice(&slice_coords);
            let sum_value = slice.iter().sum::<T>();
            new_data.push(sum_value);
        }

        let new_tensor_shape = RowMajorTensorShape::create_tensor_shape(&result_shape);
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            tensor_shape: new_tensor_shape,
            vec_len: new_data.len(),
        }
    } */

}



impl<T:Copy + std::fmt::Debug + std::ops::Sub<Output = T> + std::ops::Div<Output = T>> TensorTraits<T> for Tensor<T> 
    where T: Float,
     {
	
    fn sub(&self, other: &Self) -> Self {
            self.broadcasted_apply(other, |a, b| a.sub(&b), false)
        }

    fn div(&self, other: &Self) -> Self {
            self.broadcasted_apply(other, |a, b| a.div(&b), false)
        }

     
    fn log(&self) -> Self {
        self._elementwise(|x| {
            x.ln()
        })
    }

    fn pow(&self, n: T) -> Self {
        self._elementwise2(|x, n| {
            x.powf(n)
        },n)
    }
    
    fn sum(&self, axes: &[usize]) {
        self.collapse(Add::add, axes)
    }

}

impl<T> Clone for Tensor<T> where 
    Vec<T>: FromIterator<T>,
    T: Float,
    {
		fn clone(&self) -> Self {
				Tensor {
					data: Arc::clone(&self.data),              
					tensor_shape: self.tensor_shape.clone(),       
					vec_len: self.vec_len,
				}
			}
}

pub type F32Type = Tensor<f32>;

#[cfg(test)]
mod tests {

	use super::*;

    #[test]
    fn tests_new_raw() {
        let _r = Tensor::<f32>::new_raw(&[2,2], vec![1., 2., 3., 6.]);
        assert_eq!(_r.shape(), &[2,2]);
        assert_eq!(_r.strides(), &[2,1]);
        let _data  = _r.data.read().unwrap();
        assert_eq!(*_data, vec![1., 2., 3., 6.]);
        let _r2 = Tensor::<f64>::new_raw(&[2,2], vec![1., 2., 3., 6.]);
        assert_eq!(_r2.shape(), &[2,2]);
        assert_eq!(_r2.strides(), &[2,1]);
        let _data  = _r2.data.read().unwrap();
        assert_eq!(*_data, vec![1., 2., 3., 6.]);
    }

    #[test]
    fn test_zip() {
        let _r = Tensor::<f32>::new_raw(&[2,2], vec![1., 2., 3., 6.0]);
        let _r2 = Tensor::<f32>::new_raw(&[2,2], vec![0., 5., 3., 7.0]);
        let _e = _r.zip(&_r2, |x,y| x - y);
        assert_eq!(_e.shape(), &[2,2]);
        assert_eq!(_e.strides(), &[2,1]);
        let _data  = _e.data.read().unwrap();
        assert_eq!(*_data, vec![1.0, -3.0, 0.0, -1.0]);   
    }

    #[test]
    fn tests_ravel() {
        let _r2 = Tensor::<f32>::new_raw(&[2,2], vec![0., 5., 3., 7.0]);
        let _data  = _r2.data.read().unwrap();
        assert_eq!(*_data, _r2.ravel());
    }

    #[test]
    fn tests_tensor_shape() {
        let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.shape(), &[2, 3]);
    }


	fn tests_reshape1(_orig_shape: &[usize], _new_shape: &[usize], _expected_strides: &[usize]) {
		let _avec: Vec<_> = (0..24_u16).map(f32::from).collect();
		let _tensor = Tensor::<f32>::new_raw(&[24], _avec);

		let _tensor2 = _tensor.reshape(_new_shape);
		
        assert_eq!(_tensor2.tensor_shape.shape, _new_shape);
		assert_eq!(_tensor2.tensor_shape.strides, _expected_strides);
        let _tensor3 = _tensor.reshape2(_new_shape);
		assert_eq!(_tensor3.tensor_shape.shape, _new_shape);
		assert_eq!(_tensor3.tensor_shape.strides, _expected_strides);
	}

	#[test]
	fn tests_reshape() {
		tests_reshape1(&[24], &[3, 2, 4], &[8, 4, 1]);
		tests_reshape1(&[2, 1, 3, 1, 4], &[2, 3, 4], &[12, 4, 1]);
		tests_reshape1(&[2, 1, 3, 1, 4], &[2, 3, 4, 1], &[12, 4, 1, 1]);	
	}    

	#[test]
	fn tests_subtraction() {
		let a = Tensor::<f32>::new_raw(&[4, 3], vec![1., 2., 3.,2., 3., 4.,3., 4., 5.,4., 5., 6.]);
		let aa = a.reshape(&[4,1,3]);
		assert_eq!(aa.tensor_shape.shape, &[4, 1, 3]);
		assert_eq!(aa.tensor_shape.strides, &[3, 3, 1]);
		let b = Tensor::<f32>::new_raw(&[2,3], vec![4., 4., 4., 5., 5., 5.]);
		let _new = TensorTraits::sub(&aa, &b);
		assert_eq!(_new.tensor_shape.shape, &[4, 2, 3]);
		assert_eq!(_new.tensor_shape.strides, &[6, 3, 1]);
		let _new_data = _new.data.read().unwrap();
		assert_eq!(_new_data.to_vec(), vec![-3.0, -2.0, -1.0, -4.0, -3.0, -2.0, -2.0, -1.0, 0.0, -3.0, -2.0, 
											-1.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0]);

	}

    #[test]
    fn tests_div() {
        let a = Tensor::new_raw(&[2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        let b = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = a.div(&b);
        assert_eq!(result.ravel(), vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);      
    }

    #[test]
    fn tests_get_row(){
        let g = Tensor::<f32>::new_raw(&[4, 3], vec![1., 2., 3.,2., 3., 4.,3., 4., 5.,4., 5., 6.]);
        assert_eq!(g.get_row(0), &[1., 2., 3.]);
        assert_eq!(g.get_row(1), &[2., 3., 4.]);
        assert_eq!(g.get_row(2), &[3., 4., 5.]);
        assert_eq!(g.get_row(3), &[4., 5., 6.]);
        let ga = g.reshape(&[4,1,3]);
        assert_eq!(ga.get_row(0), &[1., 2., 3.]);
        assert_eq!(ga.get_row(1), &[2., 3., 4.]);
        assert_eq!(ga.get_row(2), &[3., 4., 5.]);
        assert_eq!(ga.get_row(3), &[4., 5., 6.]);
        let other1 = Tensor::<f32>::new_raw(&[2,3], vec![4., 4., 4., 5., 5., 5.]);
	    assert_eq!(other1.get_row(0), &[4., 4., 4.]);
	    assert_eq!(other1.get_row(1), &[5., 5., 5.]);
        let new1 = TensorTraits::sub(&ga, &other1);
	    
    	assert_eq!(new1.get_row(0), &[-3.0, -2.0, -1.0, -4.0, -3.0, -2.0]);
	    assert_eq!(new1.get_row(1), &[-2.0, -1.0, 0.0, -3.0, -2.0, -1.0]);

    }

    #[test]
    fn test_set_row(){
        let _r1 = Tensor::<f32>::new_raw(&[2,2], vec![1., 1., 1., 1.0]);
        _r1.set_row(0, &[2., 2.]);
        _r1.set_row(1, &[4., 4.]);
        let _new_data = _r1.data.read().unwrap();
        assert_eq!(*_new_data, &[2.0, 2.0, 4.0, 4.0]);

    }

    #[test]
    fn test_ops() {
        let _a = Tensor::<f32>::new_raw(&[2, 2], vec![1., 2., 3., 4. ]);
        let _log = _a.log();
        let _new_data = _log.data.read().unwrap();
        assert_eq!(*_new_data, &[0.0, 0.6931472, 1.0986123, 1.3862944]);
        let _pow = _a.pow(2.);
        let _new_data = _pow.data.read().unwrap();
        assert_eq!(*_new_data, &[1.0, 4.0, 9.0, 16.0]);
        
    }

    #[test]
    fn tests_with_contiguous_data() {
        let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let new_tensor = tensor.with_contiguous_data(vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
	    assert_eq!(new_tensor.ravel(), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn tests_with_tensor_shape() {
        let tensor = Tensor::new_raw(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let new_shape = RowMajorTensorShape::create_tensor_shape(&[2, 3]);
        let reshaped_tensor = tensor.with_tensor_shape(new_shape);
        assert_eq!(reshaped_tensor.shape(), &[2, 3]);
    }

    #[test]
    fn tests_strides() {
        let tensor = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.strides(), &[3, 1]);
    }

    #[test]
    fn tests_broadcasted_apply() {
        let a = Tensor::new_raw(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new_raw(&[3], vec![1.0, 2.0, 3.0]);
        let result = a.broadcasted_apply(&b, |a, b| a.sub(&b), false);
        assert_eq!(result.ravel(), vec![0.0, 0.0, 0.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn tests_expanded() {
        let tensor = Tensor::new_raw(&[1, 3], vec![1.0, 2.0, 3.0]);
        let expanded = tensor.expand(&[2, 3]);
        assert_eq!(expanded.shape(), &[2, 3]);
    }


}

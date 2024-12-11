//use num_traits::PrimInt;
/// Generates a vector of ranges for each element in the input vector.
///
/// This function takes a vector of `usize` elements and, for each element,
/// creates a range starting from `0` up to (but not including) the element.
/// Each range is collected into a vector, resulting in a vector of vectors.
///
/// # Arguments
/// - `input_vector`: A reference to a vector of `usize` values. Each value
///   determines the upper bound (exclusive) of the corresponding range.
///
/// # Returns
/// A vector of vectors, where each inner vector represents a range of values
/// from `0` to the corresponding element in `input_vector` (exclusive).
///
/// # Example
/// ```
/// use tensor::numeric::explode_ranges_from_vector_elements;
/// 
/// let input = vec![3, 2, 4];
/// let result = explode_ranges_from_vector_elements(&input);
/// assert_eq!(result, vec![vec![0, 1, 2], vec![0, 1], vec![0, 1, 2, 3]]);
/// ```
pub fn explode_ranges_from_vector_elements(input_vector: &Vec<usize>) -> Vec<Vec<usize>>{ 
    input_vector.iter().map(| &i | (0..i).collect::<Vec<_>>()).collect::<Vec<_>>()
}

/// Generates a vector of ranges for each element in the input vector.
///
/// Similar to `explode_ranges_from_vector_elements`, this function creates
/// ranges for each element in the input vector. It differs only in that the
/// input is a vector of `i32` values.
///
/// # Arguments
/// - `input_vector`: A reference to a vector of `i32` values. Each value
///   determines the upper bound (exclusive) of the corresponding range.
///
/// # Returns
/// A vector of vectors, where each inner vector represents a range of values
/// from `0` to the corresponding element in `input_vector` (exclusive).
///
/// # Example
/// ```
/// use tensor::numeric::explode_ranges_from_vector_elements2;
/// 
/// let input = vec![3, 2, 4];
/// let result = explode_ranges_from_vector_elements2(&input);
/// assert_eq!(result, vec![vec![0, 1, 2], vec![0, 1], vec![0, 1, 2, 3]]);
/// ```
pub fn explode_ranges_from_vector_elements2(input_vector: &Vec<i32>) -> Vec<Vec<i32>>{ 
    input_vector.iter().map(| &i | (0..i).collect::<Vec<_>>()).collect::<Vec<_>>()
}

/// Computes the Cartesian product of ranges derived from the input vector.
///
/// This function generates ranges from each element in the input vector,
/// where each range starts at `0` and ends at the element's value (exclusive).
/// The Cartesian product is then computed from these ranges, producing all
/// possible combinations of elements.
///
/// # Arguments
/// - `input_vector`: A reference to a vector of `usize` values. Each value
///   determines the upper bound (exclusive) of the corresponding range.
///
/// # Returns
/// A vector of vectors, where each inner vector represents one combination
/// of the Cartesian product of the ranges generated from `input_vector`.
///
/// # Example
/// ```
/// use tensor::numeric::cartesian_product_from_shape;
/// 
/// let input = vec![3, 2];
/// let result = cartesian_product_from_shape(&input);
/// assert_eq!(result, vec![
///     vec![0, 0], vec![0, 1],
///     vec![1, 0], vec![1, 1],
///     vec![2, 0], vec![2, 1]
/// ]);
/// ```
pub fn cartesian_product_from_shape(input_vector: &Vec<usize>) -> Vec<Vec<usize>> {
    let exploded_ranges = explode_ranges_from_vector_elements(&input_vector);
    //println!("{:?}", cartesian_product(&exploded_ranges));
    cartesian_product(&exploded_ranges)
}

/// Computes the Cartesian product of ranges derived from the input vector.
///
/// Similar to `cartesian_product_from_shape`, this function operates on
/// an input vector of `i32` values instead of `usize`. Each element in the
/// input vector specifies the upper bound (exclusive) for a range starting
/// at `0`. The Cartesian product is then computed from these ranges.
///
/// # Arguments
/// - `input_vector`: A reference to a vector of `i32` values. Each value
///   determines the upper bound (exclusive) of the corresponding range.
///
/// # Returns
/// A vector of vectors, where each inner vector represents one combination
/// of the Cartesian product of the ranges generated from `input_vector`.
///
/// # Example
/// ```
/// use tensor::numeric::cartesian_product_from_shape2;
/// 
/// let input = vec![3, 2];
/// let result = cartesian_product_from_shape2(&input);
/// assert_eq!(result, vec![
///     vec![0, 0], vec![0, 1],
///     vec![1, 0], vec![1, 1],
///     vec![2, 0], vec![2, 1]
/// ]);
/// ```
pub fn cartesian_product_from_shape2(input_vector: &Vec<i32>) -> Vec<Vec<i32>> {
    let exploded_ranges = explode_ranges_from_vector_elements2(&input_vector);
    cartesian_product(&exploded_ranges)
}

/// Computes the Cartesian product of a slice of vectors.
///
/// This function takes a slice of vectors and computes the Cartesian product
/// of all their elements. The resulting vector contains all possible
/// combinations, where each combination is represented as a vector.
///
/// # Arguments
/// - `input_vectors`: A slice of vectors, where each vector represents a set
///   of elements to combine.
///
/// # Returns
/// A vector of vectors, where each inner vector represents one combination
/// from the Cartesian product.
///
/// # Example
/// ```
/// use tensor::numeric::cartesian_product;
/// 
/// let input = vec![vec![0, 1], vec![2, 3]];
/// let result = cartesian_product(&input);
/// assert_eq!(result, vec![
///     vec![0, 2], vec![0, 3],
///     vec![1, 2], vec![1, 3]
/// ]);
/// ```
///
/// # Panics
/// This function will not panic, even if the input slice is empty.
pub fn cartesian_product<T>(input_vectors: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Clone,
{
    if input_vectors.is_empty() {
        return vec![vec![]];
    }

    let mut result = Vec::new();
    let first_vector = &input_vectors[0];
    let remaining_vectors = &input_vectors[1..];

    for item in first_vector {
        let mut combinations = cartesian_product(remaining_vectors);
        for combination in &mut combinations {
            combination.insert(0, item.clone());
        }
        result.extend(combinations);
    }

    result
}


#[cfg(test)]
mod tests {

	use super::*;

    #[test]
    fn tests_explode_ranges() {
        let ex = explode_ranges_from_vector_elements(&vec![2,1,3]);
        assert_eq!(ex, vec![vec![0, 1], vec![0], vec![0, 1, 2]]);
        let ex2 = explode_ranges_from_vector_elements2(&vec![2,1,3]);
        assert_eq!(ex2, vec![vec![0, 1], vec![0], vec![0, 1, 2]]);
    }

    #[test]
	fn tests_cartesian_product_from_shape() {
        let c1 = cartesian_product_from_shape(&vec![2,2,2]);
        assert_eq!(c1, &[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]);
        let c2 = cartesian_product_from_shape(&vec![1,2,2]);
        assert_eq!(c2, &[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]);
        let c3 = cartesian_product_from_shape(&vec![4,2,3]);
        assert_eq!(c3, &[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1, 0, 1], 
                        [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], 
                        [2, 1, 1], [2, 1, 2], [3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 1, 0], [3, 1, 1], [3, 1, 2]]);

    }

    #[test]
    fn tests_cartesian_product_from_shape2() {
        let c1 = cartesian_product_from_shape2(&vec![2,2,2]);
        assert_eq!(c1, &[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]);
        let c2 = cartesian_product_from_shape2(&vec![1,2,2]);
        assert_eq!(c2, &[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]);
        let c3 = cartesian_product_from_shape2(&vec![4,2,3]);
        assert_eq!(c3, &[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1, 0, 1], 
                        [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], 
                        [2, 1, 1], [2, 1, 2], [3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 1, 0], [3, 1, 1], [3, 1, 2]]);

    }

    #[test]
    fn tests_cartesian_product() {
        let input = vec![vec![0, 1], vec![2, 3]];
        let result = cartesian_product(&input);
        assert_eq!(result, vec![
        vec![0, 2], vec![0, 3],
        vec![1, 2], vec![1, 3]
        ]);
    }


}
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub stride: Vec<usize>,
}

pub struct TensorView<'a> {
    pub shape: Vec<usize>,
    pub data: &'a [f32],
    pub stride: Vec<usize>,
    pub offset: usize,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, String> {
        let result = validate_tensor(&shape, data.len());
        if result {
            let stride = Tensor::calculate_stride(&shape);
            Ok(Self {
                shape,
                data,
                stride,
            })
        } else {
            Err(String::from("error"))
        }
    }

    pub fn calculate_stride(shape: &Vec<usize>) -> Vec<usize> {
        let n_dim = shape.len();
        let mut slice_vec = vec![];

        for i in 0..n_dim {
            let mut product = 1;

            for j in (i + 1)..n_dim {
                product = product * shape[j];
            }
            slice_vec.push(product)
        }

        return slice_vec;
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let result = validate_tensor(&new_shape, self.len());
        if result {
            let new_stride = Tensor::calculate_stride(&new_shape);
            Ok({
                self.shape = new_shape;
                self.stride = new_stride;
            })
        } else {
            Err(String::from("Cannot reshape, not valid."))
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> TensorView<'a> {
    pub fn new(view_parent: &'a Tensor) -> Self {
        Self {
            shape: view_parent.shape.clone(),
            data: &view_parent.data,
            stride: view_parent.stride.clone(),
            offset: 0,
        }
    }
}

pub fn calculate_index(tensor: &Tensor, coord: &[usize]) -> Result<usize, String> {
    let shape = tensor.shape();
    let n_dim = tensor.ndim();
    let slice_vec = tensor.stride();
    let mut index: usize = 0;

    if coord.len() != n_dim {
        return Err("Coord Dimensions != Tensor Dimensions".to_string());
    }

    for i in 0..n_dim {
        // Coord must be less than than shape
        if coord[i] >= shape[i] {
            return Err("Coords out of Bounds".to_string());
        }
    }

    for i in 0..slice_vec.len() {
        index += slice_vec[i] * coord[i];
    }

    return Ok(index);
}

pub fn get(tensor: &Tensor, index: usize) -> f32 {
    return tensor.data[index];
}

fn count_elements(shape: &[usize]) -> Option<usize> {
    // Get expected length for certain shape
    let mut elements: usize = 1;

    for &i in shape {
        let result = i.checked_mul(elements);

        match result {
            Some(value) => {
                elements = value;
            }
            None => return None,
        }
    }

    return Some(elements);
}

pub fn validate_tensor(shape: &[usize], length: usize) -> bool {
    // Validate the tensor. Do not need ownership of the values, just borrow.

    let elements = count_elements(shape);
    match elements {
        Some(value) => return value == length,
        None => return false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_tensor_check() {
        assert_eq!(validate_tensor(&[0], 1), false);
        assert_eq!(validate_tensor(&[1, 1, 1, 1], 1), true);
        assert_eq!(validate_tensor(&[], 1), true);
        assert_eq!(validate_tensor(&[], 0), false);
    }
    #[test]
    fn count_elements_check() {
        assert_eq!(count_elements(&[usize::MAX]), Some(usize::MAX));
        assert_eq!(count_elements(&[usize::MAX, 2]), None);
        assert_eq!(count_elements(&[usize::MAX, 0]), Some(0));
    }
    #[test]
    fn struct_check() {
        let t_1 = Tensor::new(vec![2, 3], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect("Error making tensor");
        assert_eq!(t_1.shape(), &[2, 3]);
        assert_eq!(t_1.len(), 6);
        assert_eq!(t_1.ndim(), 2);
        assert_eq!(t_1.is_empty(), false);
    }
    #[test]
    fn index_check_1() {
        let t_1 = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 3.0]).expect("Error making tensor");
        assert_eq!(calculate_index(&t_1, &[2]).expect("Error"), 2);
        assert_eq!(get(&t_1, 2), 3.0);
    }
    #[test]
    fn index_check_2() {
        let t_1 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Error making tensor");
        assert_eq!(calculate_index(&t_1, &[1, 2]).expect("Error"), 5);
        assert_eq!(get(&t_1, 5), 6.0);
    }
    #[test]
    fn index_check_3() {
        let v: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t_1 = Tensor::new(vec![2, 3, 4], v).expect("Error making tensor");
        assert_eq!(calculate_index(&t_1, &[1, 2, 3]).expect("Error"), 23);
        assert_eq!(get(&t_1, 23), 24.0);
    }
    #[test]
    fn index_check_4() {
        let t_1 = Tensor::new(vec![], vec![1.0]).expect("Error making tensor");
        assert_eq!(calculate_index(&t_1, &[]).expect("Error"), 0);
        assert_eq!(get(&t_1, 0), 1.0);
    }
    #[test]
    fn index_check_5() {
        let t_1 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Error making tensor");
        assert!(calculate_index(&t_1, &[1]).is_err());
    }
}

#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sys/stat.h>
#include <chrono>
#include "D:/Code/eigen-3.4.0/Eigen/Dense"


//eigen initializations
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tensor3D = std::vector<Matrix>;


bool is_directory(const std::string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0)
        return false;
    return true;
}


// simple layer class for other network layers to inherit from.
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient, double learning_rate) = 0;
};

//denselayer
class Dense : public Layer {
private:
    Matrix wts;
    Vector bias;
    Matrix input;
    double gradient_clip_value = 5.0;  
    double weight_decay = 0.0001;
    
    //clipping gradients helped with the output returning nans, so it is included
    double clip_gradient(double value) {
        return std::max(-gradient_clip_value, std::min(gradient_clip_value, value));
    }

public:
    Dense(int in_size, int output_size) {
        double std_dev = std::sqrt(2.0 / (in_size + output_size));
        
        std:: random_device rd;
        std::mt19937 gen(rd());
        std :: normal_distribution<> d(0, std_dev);
        
        wts = Matrix(output_size , in_size);
        bias = Vector::Zero(output_size);
        
        //initializing a normal distribution of wts for dense layer
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < in_size; j++) {
                wts(i, j) = d(gen);
            }
        }

    }
    
    Matrix forward(const Matrix& input) override {
        this->input = input;
        Matrix output = wts * input + bias.replicate(1, input.cols());
        
        // checking for nans
        for (int i = 0; i < output.rows(); i++) {
            for (int j = 0; j < output.cols(); j++) {
                if (std::isnan(output(i, j)) || std::isinf(output(i, j))) {

                    output(i, j) = 0.0;
                }
            }
        }
        
        return output;
    }
    
    Matrix backward(const Matrix& output_gradient, double learning_rate) override {
        Matrix clipped_gradient = output_gradient;
        for (int i = 0; i < clipped_gradient.rows(); i++) {


            for (int j = 0; j < clipped_gradient.cols(); j++) {

                if (std::isnan(clipped_gradient(i, j)) || std::isinf(clipped_gradient(i, j))) {
                    clipped_gradient(i, j) = 0.0;
                } else {
                    clipped_gradient(i, j) = clip_gradient(clipped_gradient(i, j));
                }
            }
        }
        
        //weight step
        Matrix wts_gradient = clipped_gradient * input.transpose();

        wts_gradient += weight_decay * wts;
        

        for (int i = 0; i < wts_gradient.rows(); i++) {
            for (int j = 0; j < wts_gradient.cols(); j++) {
                if (std::isnan(wts_gradient(i, j)) || std::isinf(wts_gradient(i, j))) {
                    wts_gradient(i, j) = 0.0; 
                } else {
                    wts_gradient(i, j) = clip_gradient(wts_gradient(i, j));
                }
            }
        }
        
        //updating the model wts according to the formula
        wts -= learning_rate * wts_gradient;
        
        Vector bias_gradient = clipped_gradient.rowwise().sum();
        

        for (int i = 0; i < bias_gradient.size(); i++) {
            if (std::isnan(bias_gradient(i)) || std::isinf(bias_gradient(i))) {
                bias_gradient(i) = 0.0;
            } else {
                bias_gradient(i) = clip_gradient(bias_gradient(i));
            }
        }
        
        bias -= learning_rate * bias_gradient;
        
        //computing input gradient according to the formula
        Matrix in_gradient = wts.transpose() * clipped_gradient;
        
        for (int i = 0; i < in_gradient.rows(); i++) {
            for (int j = 0; j < in_gradient.cols(); j++) {
                if (std::isnan(in_gradient(i, j)) || std::isinf(in_gradient(i, j))) {
                    in_gradient(i, j) = 0.0;
                } else {
                    in_gradient(i, j) = clip_gradient(in_gradient(i, j));
                }
            }
        }
        
        return in_gradient;
    }
};


//base class for activation layer, ReLU inherits from this
class Activation : public Layer {
protected:
    std::function<Matrix(const Matrix&)> activation_function;
    std::function<Matrix(const Matrix&)> activation_derivative;
    Matrix input;
    
public:
    Activation(std::function<Matrix(const Matrix&)> activation_function,
               std::function<Matrix(const Matrix&)> activation_derivative) 
        : activation_function(activation_function), activation_derivative(activation_derivative) {}
    
    Matrix forward(const Matrix& input) override {
        this->input = input;
        return activation_function(input);
    }
    
    Matrix backward(const Matrix& output_gradient, double learning_rate) override {
        return (output_gradient.array() * activation_derivative(input).array()).matrix();
    }
};

class Sigmoid : public Activation {
public:
    Sigmoid() : Activation(
        // Sigmoid function
        [](const Matrix& x) -> Matrix {
            return (1.0 / (1.0 + (-x.array()).exp())).matrix();
        },
        // Sigmoid derivative
        [](const Matrix& x) -> Matrix {
            Matrix s = (1.0 / (1.0 + (-x.array()).exp())).matrix();
            return (s.array() * (1.0 - s.array())).matrix();
        }
    ) {}
};

//implementation of correlate using eigen.
//correlation is used in the forward and backward pass of convolution layer
Matrix correlate2d(const Matrix& input, const Matrix& kernel) {
    int rows, cols;
    rows = input.rows() - kernel.rows() + 1;
    cols = input.cols() - kernel.cols() + 1;

    if (rows <= 0 || cols <= 0) {
        return Matrix::Zero(1, 1);
    }

    Matrix output = Matrix::Zero(rows, cols);

    for (int i = 0; i <= input.rows() - kernel.rows(); i++) {
        for (int j = 0; j <= input.cols() - kernel.cols(); j++) {
            Matrix window = input.block(i, j, kernel.rows(), kernel.cols());
            //take sub-matrix of input of size equal to kernel and do point-by-point multiplication with kernel
            double sum = (window.array() * kernel.array()).sum();
            output(i, j) = sum;
        }
    }

    return output;
}

Matrix convolve2d(const Matrix& input, const Matrix& kernel) {    
    int rows, o_cols;

    rows = input.rows() + kernel.rows() - 1;
    o_cols = input.cols() + kernel.cols() - 1;

    
    Matrix output = Matrix::Zero(rows, o_cols);
    

    //padding the matrix for full convolution
    Matrix padded = Matrix::Zero(input.rows() + 2*(kernel.rows()-1), input.cols() + 2*(kernel.cols()-1));
    padded.block(kernel.rows()-1, kernel.cols()-1, input.rows(), input.cols()) = input;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < o_cols; j++) {
            if (i + kernel.rows() <= padded.rows() && j + kernel.cols() <= padded.cols()) {
                //following the formula
                Matrix window = padded.block(i, j, kernel.rows(), kernel.cols());
                double sum = (window.array() * kernel.array()).sum();
                output(i, j) = sum;
            
            }
        }
    }
    
    return output;
}


class Convolutional : public Layer {
private:
    std::vector<std::vector<Matrix>> kernels;  
    std::vector<Matrix> biases;                
    std::vector<Matrix> input;
    int in_depth, kernelnum, kernel_size;
    int in_height, in_width;
    
public:
    Convolutional(int in_depth, int in_height, int in_width, int kernel_size, int kernelnum) 
        : in_depth(in_depth), kernelnum(kernelnum), kernel_size(kernel_size),
          in_height(in_height), in_width(in_width) {
        

        double std_dev = std::sqrt(2.0 / (in_depth * kernel_size * kernel_size + kernelnum * kernel_size * kernel_size));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, std_dev);
        
        //kernel initialization
        kernels.resize(kernelnum);
        for (int i = 0; i < kernelnum; i++) {
            kernels[i].resize(in_depth);
            for (int j = 0; j < in_depth; j++) {
                kernels[i][j] = Matrix(kernel_size, kernel_size);
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        //kernels[i][j](k, l) = d(gen);
                        if (i!=1){
                            kernels[i][j](k, l)=16000;
                        }
                        else{
                            kernels[i][j](k, l)=-16000;
                        }
                    }
                }
            }
        }
        
        int output_height = in_height - kernel_size + 1;
        int output_width = in_width - kernel_size + 1;
        
        biases.resize(kernelnum);
        for (int i = 0; i < kernelnum; i++) {
            biases[i] = Matrix::Zero(output_height, output_width);
        }
    }

    std::vector<Matrix> reshapeInput(const Matrix& in_matrix) {
        std::vector<Matrix> result(in_depth);
        
        
        if (in_matrix.rows() == in_depth * in_height * in_width && in_matrix.cols() == 1) {
            for (int d = 0; d < in_depth; d++) {
                result[d] = Matrix(in_height, in_width);
                for (int i = 0; i < in_height; i++) {
                    for (int j = 0; j < in_width; j++) {

                        int index = d * in_height * in_width + i * in_width   + j;
                        result[d](i, j) =  in_matrix(index, 0);
                    }
                }
            }
        } else {
            std::cout<<"incorrect  dimensions found";
            
            for (int d = 0; d < in_depth; d++) {
                result[d] = Matrix :: Zero(in_height, in_width);
            }
        }
        
        return result;
    }

    Matrix reshapeOutput(const std::vector<Matrix>& matrices) {
        if (matrices.empty()) {
            return Matrix::Zero(1, 1);
        }
        
        int height = matrices[0].rows();
        int width = matrices[0].cols();
        int total_size = matrices.size() * height * width;
        
        Matrix result = Matrix::Zero(total_size, 1);
        
        for (size_t d = 0; d < matrices.size(); d++) {
            if (matrices[d].rows() != height || matrices[d].cols() != width) {
                printf("Wrong dimensions\n");
                return result; 
            }
            
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int index = d * height * width + i * width + j;
                    if (index < result.rows()) {
                        result(index, 0) = matrices[d](i, j);
                    } else {
                        std::cout << "could not reshape correctly";
                    }
                }
            }
        }
        
        return result;
    }
    
    Matrix forward(const Matrix& in_matrix) override {
        input = reshapeInput(in_matrix);
        int output_height = in_height - kernel_size + 1;
        int output_width = in_width - kernel_size + 1;
        
        std::vector<Matrix> output = biases;

        for (int i = 0; i < kernelnum; i++) {
            for (int j = 0; j < in_depth; j++) {
                //formula for forward pass in the convolutional layer
                Matrix corr = correlate2d(input[j], kernels[i][j]);
                output[i] += corr;
            }
        }
        
        return reshapeOutput(output);
    }
    
    Matrix backward(const Matrix& output_gradient, double learning_rate) override {
        if (input.empty()) {
            std::cout<<"Empty input, returning 0s";
            return Matrix::Zero(in_depth * in_height * in_width, 1);
        }
        
        int output_height = in_height - kernel_size + 1;
        int output_width = in_width - kernel_size + 1;
        
        std::vector<Matrix> output_gradient_tensor(kernelnum);
        
        for (int d = 0; d < kernelnum; d++) {
            output_gradient_tensor[d] = Matrix::Zero(output_height, output_width);
            for (int i = 0; i < output_height; i++) {
                for (int j = 0; j < output_width; j++) {
                    int index = d * output_height * output_width + i * output_width + j;
                    if (index < output_gradient.rows()) {
                        output_gradient_tensor[d](i, j) = output_gradient(index, 0);
                    }
                }
            }
        }

        //creating gradients necessary for the backprop calculations
        std::vector<std::vector<Matrix>> kernels_gradient(kernelnum, std::vector<Matrix>(in_depth));
        std::vector<Matrix> in_gradient(in_depth);
        
        for (int j = 0; j < in_depth; j++) {
            in_gradient[j] = Matrix::Zero(in_height, in_width);
        }
        
        for (int i = 0; i < kernelnum; i++) {
            for (int j = 0; j < in_depth; j++) {

                //kernel gradient formula: dError/dKernal[i][j]=input[j] correlated with dError/dOutput

                kernels_gradient[i][j] = correlate2d(input[j], output_gradient_tensor[i]);
                
                //full convolution step
                Matrix flipped_kernel(kernel_size, kernel_size);
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        flipped_kernel(k, l) = kernels[i][j](kernel_size - 1 - k, kernel_size - 1 - l);
                    }
                }
                //formula for input update step: dError/dInput=sum(dError/dOutput fully convolved with Kernel)
                Matrix full_conv = convolve2d(output_gradient_tensor[i], flipped_kernel);
                
                if (full_conv.rows() == in_gradient[j].rows() && full_conv.cols() == in_gradient[j].cols()) {
                    in_gradient[j] += full_conv;
                } else {
                    std::cout << "error with the full convolution dimensions ";
                }
            }
        }
        
        //kernel update
        for (int i = 0; i < kernelnum; i++) {
            for (int j = 0; j < in_depth; j++) {
                if (kernels_gradient[i][j].rows() == kernels[i][j].rows() && 
                    kernels_gradient[i][j].cols() == kernels[i][j].cols()) {
                    kernels[i][j] -= learning_rate * kernels_gradient[i][j];
                }
            }
        }
        
        //biasformula: dError/dBias=dError/dOutput
        for (int i = 0; i < kernelnum; i++) {
            if (output_gradient_tensor[i].rows() == biases[i].rows() && 
                output_gradient_tensor[i].cols() == biases[i].cols()) {
                biases[i] -= learning_rate * output_gradient_tensor[i];
            }
        }
        return reshapeOutput(in_gradient);
    }
};

class MaxPooling : public Layer {
private:
    int pool_size;
    int stride;
    std::vector<Matrix> input;
    std::vector<Matrix> max_indices;
    int in_depth;
    int in_height;
    int in_width;

public:
    MaxPooling(int in_depth, int in_height, int in_width, int pool_size = 2, int stride = 2)
        : in_depth(in_depth), in_height(in_height), in_width(in_width),
          pool_size(pool_size), stride(stride) {}

    std::vector<Matrix> reshapeInput(const Matrix& in_matrix) {
        std::vector<Matrix> result(in_depth);
        
        if (in_matrix.rows() == in_depth * in_height * in_width && in_matrix.cols() == 1) {
            for (int d = 0; d < in_depth; d++) {
                result[d] = Matrix(in_height, in_width);
                for (int i = 0; i < in_height; i++) {
                    for (int j = 0; j < in_width; j++) {
                        int index = d * in_height * in_width + i * in_width + j;
                        result[d](i, j) = in_matrix(index, 0);
                    }
                }
            }
        } else {
            printf("Wrong dimensions in maxpooling");
            for (int d = 0; d < in_depth; d++) {
                result[d] = Matrix::Zero(in_height, in_width);
            }
        }
        
        return result;
    }

    Matrix reshapeOutput(const std::vector<Matrix>& matrices) {
        if (matrices.empty()) {
            return Matrix::Zero(1, 1);
        }
        
        int out_height = matrices[0].rows();
        int out_width = matrices[0].cols();
        int total_size = matrices.size() * out_height * out_width;
        
        Matrix result = Matrix::Zero(total_size, 1);
        
        for (size_t d = 0; d < matrices.size(); d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    int index = d * out_height * out_width + i * out_width + j;
                    if (index < result.rows()) {
                        result(index, 0) = matrices[d](i, j);
                    }
                }
            }
        }
        
        return result;
    }

    Matrix forward(const Matrix& in_matrix) override {
        input = reshapeInput(in_matrix);
        std::vector<Matrix> output(in_depth);
        max_indices.resize(in_depth);
        
        for (int d = 0; d < in_depth; d++) {
            int out_height = (in_height - pool_size) / stride + 1;
            int out_width = (in_width - pool_size) / stride + 1;
            
            output[d] = Matrix::Zero(out_height, out_width);
            max_indices[d] = Matrix::Zero(out_height, out_width);
            
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_idx = 0;
                    
                    for (int pi = 0; pi < pool_size; pi++) {
                        for (int pj = 0; pj < pool_size; pj++) {
                            double val = input[d](i * stride + pi, j * stride + pj);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = pi * pool_size + pj;
                            }
                        }
                    }
                    
                    output[d](i, j) = max_val;
                    max_indices[d](i, j) = max_idx;
                }
            }
        }
        
        return reshapeOutput(output);
    }

    Matrix backward(const Matrix& output_gradient, double learning_rate) override {
        std::vector<Matrix> grad_input(in_depth);
        
        for (int d = 0; d < in_depth; d++) {
            grad_input[d] = Matrix::Zero(in_height, in_width);
            
            int out_height = (in_height - pool_size) / stride + 1;
            int out_width = (in_width - pool_size) / stride + 1;
            
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    int max_idx = static_cast<int>(max_indices[d](i, j));
                    int pi = max_idx / pool_size;
                    int pj = max_idx % pool_size;
                    
                    int index = d * out_height * out_width + i * out_width + j;
                    if (index < output_gradient.rows()) {
                        grad_input[d](i * stride + pi, j * stride + pj) = output_gradient(index, 0);
                    }
                }
            }
        }
        
        return reshapeOutput(grad_input);
    }
};

class ReLU : public Activation {
    public:
        ReLU() : Activation(
            //leaky relu is used instead of normal relu as it provides better results
            [](const Matrix& x) -> Matrix {
                double alpha = 0.01;
                return (x.array() > 0.0).select(x, alpha * x);
            },
            [](const Matrix& x) -> Matrix {
                double alpha = 0.01; 
                return (x.array() > 0.0).cast<double>().select(Matrix::Ones(x.rows(), x.cols()), alpha * Matrix::Ones(x.rows(), x.cols()));
            }
        ) {}
    };

//softmax activation function: used at the output to figure out which neuron to fire for the classification problem of 0-9 digits
class Softmax : public Activation {
    public:
        Softmax() : Activation(
            [](const Matrix& x) -> Matrix {
                Matrix result = Matrix::Zero(x.rows(), x.cols());
                //softmax simply finds the max value and activates that neuron
                for (int col = 0; col < x.cols(); col++) {
                    double max_val = x.col(col).maxCoeff();
                    
                    // Check for extremely large values that might cause overflow
                    if (std::isinf(max_val) || std::isnan(max_val)) {
                        std::cout << "nan in softmax";
                        for (int i = 0; i < x.rows(); i++) {
                            result(i, col) = 1.0 / x.rows();
                        }
                        continue;
                    }
                    
                    Matrix exp_values = Matrix::Zero(x.rows(), 1);
                    double sum_exp = 0.0;
                    
                    for (int i = 0; i < x.rows(); i++) {
                        if (std::isnan(x(i, col)) || std::isinf(x(i, col))) {
                            exp_values(i, 0) = 0.0;
                        } else {
                            double safe_exp = std::exp(x(i, col) - max_val);
                            exp_values(i, 0) = safe_exp;
                            sum_exp += safe_exp;
                        }
                    }
                    //nan check
                    if (sum_exp < 1e-10 || std::isnan(sum_exp) || std::isinf(sum_exp)) {
                        for (int i = 0; i < x.rows(); i++) {
                            result(i, col) = 1.0 / x.rows();
                        }
                    } else {
                        for (int i = 0; i < x.rows(); i++) {
                            result(i, col) = exp_values(i, 0) / sum_exp;
                        }
                    }
                }
                return result;
            },
            [](const Matrix& x) -> Matrix {
                return Matrix::Ones(x.rows(), x.cols());
            }
        ) {
            std::cout << "\n";
        }
        
        Matrix forward(const Matrix& input) override {
            this->input = input;
            Matrix output = activation_function(input);
            
            if (output.rows() == 10 && output.cols() == 1) {
                int pred_class = 0;
                double max_val = output(0, 0);
                
                for (int i = 0; i < output.rows(); i++) {
                    if (output(i, 0) > max_val) {
                        max_val = output(i, 0);
                        pred_class = i;
                    }
                }

            }
            
            return output;
        }
        
        Matrix backward(const Matrix& output_gradient, double learning_rate) override {
            for (int i = 0; i < output_gradient.rows(); i++) {
                for (int j = 0; j < output_gradient.cols(); j++) {
                    if (std::isnan(output_gradient(i, j)) || std::isinf(output_gradient(i, j))) {
                        return Matrix::Ones(output_gradient.rows(), output_gradient.cols()) * 0.001;
                    }
                }
            }
            return output_gradient;
        }
    };
    
    
    double categorical_cross_entropy(const Matrix& y_true, const Matrix& y_pred) {
        double epsilon = 1e-10;
        double loss = 0.0;
        
        for (int i = 0; i < y_true.rows(); i++) {
            if (std::isnan(y_pred(i, 0)) || std::isinf(y_pred(i, 0))) {
                continue; // Skip this element
            }
            
            double pred_val = std::max(epsilon, std::min(1.0 - epsilon, y_pred(i, 0)));
            if (y_true(i, 0) > 0) { 
                loss -= y_true(i, 0) * std::log(pred_val);
            }
        }
        //checking nan
        if (std::isnan(loss) || std::isinf(loss)) {
            return 10.0;
        }
        return loss;
    }
    
    Matrix categorical_cross_entropy_prime(const Matrix& y_true, const Matrix& y_pred) {
        Matrix gradient = Matrix::Zero(y_pred.rows(), y_pred.cols());
        
        for (int i = 0; i < y_pred.rows(); i++) {
            for (int j = 0; j < y_pred.cols(); j++) {
                if (std::isnan(y_pred(i, j)) || std::isinf(y_pred(i, j))) {
                    gradient(i, j) = 0.0;
                } else {
                    gradient(i, j) = y_pred(i, j) - y_true(i, j);
                    gradient(i, j) = std::max(-5.0, std::min(5.0, gradient(i, j)));
                }
            }
        }
        
        return gradient;
    }
    
    
    Matrix predict(const std::vector<Layer*>& network, const Matrix& input) {
        Matrix output = input;
        for (auto layer : network) {
            output = layer->forward(output);
        }
        return output;
    }
    
    //one_hot for data labeling
    Matrix create_one_hot(int label, int num_classes) {
        Matrix one_hot = Matrix::Zero(num_classes, 1);
        if (label >= 0 && label < num_classes) {
            one_hot(label, 0) = 1.0;
        }
        return one_hot;
    }

    void preprocess_data(
        const std::vector<Matrix>& images,
        const std::vector<int>& labels,
        std::vector<Matrix>& x_processed,
        std::vector<Matrix>& y_processed,
        int limit_per_class) {

        
        x_processed.clear();
        y_processed.clear();
        const int num_classes = 10; 
        
        double sum = 0.0;
        double sum_sq = 0.0;
        int count = 0;
        
        for (const auto& img : images) {
            for (int i = 0; i < img.rows(); i++) {
                sum += img(i, 0);
                sum_sq += img(i, 0) * img(i, 0);
                count++;
            }
        }
        
        double mean = sum / count;
        double variance = (sum_sq / count) - (mean * mean);
        double std_dev = std::sqrt(variance);
        
        if (std_dev < 1e-10) {
            std_dev = 1.0;
        }
        
        
        std::vector<int> class_counts(num_classes, 0);
        std::vector<std::vector<int>> class_indices(num_classes);
        
        //class labelling
        for (size_t i = 0; i < labels.size(); i++) {
            int label = labels[i];
            if (label >= 0 && label < num_classes) {
                class_indices[label].push_back(i);
                class_counts[label]++;
            }
        }
        
        int min_class_count = *std::min_element(class_counts.begin(), class_counts.end());
        int samples_per_class = std::min(min_class_count, limit_per_class);
        
        // std::cout << "Initial class distribution:" << std::endl;
        // for (int i = 0; i < num_classes; i++) {
        //     std::cout << "Class " << i << ": " << class_indices[i].size() << " samples" << std::endl;
        // }
        

        std::random_device rd;
        std::mt19937 g(rd());

        //takes data at random
        std::vector<int> all_indices;
        for (int i = 0; i < num_classes; i++) {
            std::shuffle(class_indices[i].begin(), class_indices[i].end(), g);
            int count_to_take = std::min(samples_per_class, static_cast<int>(class_indices[i].size()));
            for (int j = 0; j < count_to_take; j++) {
                all_indices.push_back(class_indices[i][j]);
            }
        }

        std::shuffle(all_indices.begin(), all_indices.end(), g);
        
        for (size_t i = 0; i < all_indices.size(); i++) {
            int idx = all_indices[i];
            
            if (idx < images.size()) {
                Matrix img = images[idx];
                for (int j = 0; j < img.rows(); j++) {
                    img(j, 0) = (img(j, 0) - mean) / std_dev;
                }
                x_processed.push_back(img);
                
                if (idx < labels.size()) {
                    y_processed.push_back(create_one_hot(labels[idx], num_classes));
                } else {
                    y_processed.push_back(create_one_hot(0, num_classes));  
                }
            }
        }
        
        std::vector<int> final_class_counts(num_classes, 0);
        for (size_t i = 0; i < all_indices.size(); i++) {
            int idx = all_indices[i];
            if (idx < labels.size()) {
                int label = labels[idx];
                if (label >= 0 && label < num_classes) {
                    final_class_counts[label]++;
                }
            }
        }
        
        std::cout << "Final class distribution:" << std::endl;
        for (int i = 0; i < num_classes; i++) {
            std::cout << "Class " << i << ": " << final_class_counts[i] << " samples" << std::endl;
        }
        
        //std::cout << "Preprocessing complete. Created " << x_processed.size() << " samples." << std::endl;
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        for (const auto& img : x_processed) {
            for (int i = 0; i < img.rows(); i++) {
                min_val = std::min(min_val, img(i, 0));
                max_val = std::max(max_val, img(i, 0));
            }
        }
        //std::cout << "Processed data range: [" << min_val << ", " << max_val << "]" << std::endl;
    }



void train(std::vector<Layer*>& network,
    std::function<double(const Matrix&, const Matrix&)> loss,
    std::function<Matrix(const Matrix&, const Matrix&)> loss_prime,
    const std::vector<Matrix>& x_train,
    const std::vector<Matrix>& y_train,
    int epochs = 1000,
    double learning_rate = 0.01,
    bool verbose = true)
{
    double best_error = std::numeric_limits<double>::max();
    int no_improvement_count = 0;
    const int num_classes = 10;  
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_update_time = start_time;

    for (int e = 0; e < epochs; e++)
    {
        double error = 0.0;
        std::vector<int> correct_per_class(num_classes, 0);
        std::vector<int> total_per_class(num_classes, 0);
        int total_correct = 0;

        int progress_bar_width = 50;
        int samples_processed = 0;
        int total_samples = x_train.size();
        
        if (verbose) {
            std::cout << "Epoch " << (e + 1) << "/" << epochs;

        }

        for (size_t i = 0; i < x_train.size(); i++)
        {
            Matrix output = predict(network, x_train[i]);

            error += loss(y_train[i], output);

            int pred_class = 0;
            double max_pred = output(0, 0);
            for (int j = 1; j < num_classes; j++)
            {
                if (output(j, 0) > max_pred)
                {
                    max_pred = output(j, 0);
                    pred_class = j;
                }
            }

            int true_class = 0;
            double max_true = y_train[i](0, 0);
            for (int j = 1; j < num_classes; j++)
            {
                if (y_train[i](j, 0) > max_true)
                {
                    max_true = y_train[i](j, 0);
                    true_class = j;
                }
            }

            if (true_class >= 0 && true_class < num_classes)
            {
                total_per_class[true_class]++;
                if (pred_class == true_class)
                {
                    correct_per_class[true_class]++;
                    total_correct++;
                }
            }

            Matrix grad = loss_prime(y_train[i], output);
            for (auto it = network.rbegin(); it != network.rend(); ++it)
            {
                grad = (*it)->backward(grad, learning_rate);
            }
            
            samples_processed++;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update_time).count();
            
            if (verbose && (time_diff > 200 || samples_processed == total_samples)) {
                double progress = static_cast<double>(samples_processed) / total_samples;
                int position = static_cast<int>(progress * progress_bar_width);
                
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                
                double eta = 0;
                if (progress > 0) {
                    eta = elapsed / progress - elapsed;
                }
                
                std::cout << "\r";
                std::cout << "Epoch " << (e + 1) << "/" << epochs;

                
                last_update_time = current_time;
            }
        }

        error /= x_train.size();
        double overall_accuracy = static_cast<double>(total_correct) / x_train.size() * 100;

        if (verbose)
        {
            
            auto current_time = std::chrono::high_resolution_clock::now();
            auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            

            std::cout << "\r";
            std::cout << "Epoch " << (e + 1) << "/" << epochs
                    << ", error=" << error
                    << ", accuracy=" << overall_accuracy << "%"
                    << ", time=" << epoch_time << "s" << std::endl;

            if ((e + 1) % 1 == 0 || e == 0)
            {
                std::cout << "Class accuracies:" << std::endl;
                for (int c = 0; c < num_classes; c++)
                {
                    double class_acc = total_per_class[c] > 0 ?
                        100.0 * correct_per_class[c] / total_per_class[c] : 0.0;
                    std::cout << "  Class " << c << ": " << class_acc
                            << "% (" << correct_per_class[c] << "/" << total_per_class[c] << ")" << std::endl;
                }
            }

            start_time = std::chrono::high_resolution_clock::now();
        }

        if (error < best_error)
        {
            best_error = error;
            no_improvement_count = 0;
        }
        else
        {
            no_improvement_count++;
            if (no_improvement_count >= 10)
            {
                std::cout << "Early stopping: No improvement for 10 epochs" << std::endl;
                break;
            }
        }
    }
}

void load_mnist(const std::string& images_file, const std::string& labels_file,
               std::vector<Matrix>& images, std::vector<int>& labels,
               uint32_t max_samples = std::numeric_limits<uint32_t>::max()) {

    images.clear();
    labels.clear();
    
    std::ifstream img_file(images_file, std::ios::binary);
    std::ifstream lbl_file(labels_file, std::ios::binary);
    

    uint32_t magic, num_images, num_rows, num_cols;
    img_file.read(reinterpret_cast<char*>(&magic), 4);
    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    img_file.read(reinterpret_cast<char*>(&num_rows), 4);
    img_file.read(reinterpret_cast<char*>(&num_cols), 4);
    
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
    
    

    uint32_t label_magic, num_labels;
    lbl_file.read(reinterpret_cast<char*>(&label_magic), 4);
    lbl_file.read(reinterpret_cast<char*>(&num_labels), 4);
    
    label_magic = __builtin_bswap32(label_magic);
    num_labels = __builtin_bswap32(num_labels);
    

    std::vector<uint32_t> indices(num_images);
    for (uint32_t i = 0; i < num_images; i++) {
        indices[i] = i;
    }
    
    //random sorting of the data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    uint32_t samples_to_load = std::min(num_images, max_samples);
    

    std::vector<uint8_t> all_images(num_images * num_rows * num_cols);
    std::vector<uint8_t> all_labels(num_images);
    
    img_file.read(reinterpret_cast<char*>(all_images.data()), all_images.size());
  
    lbl_file.read(reinterpret_cast<char*>(all_labels.data()), all_labels.size());
    
    for (uint32_t idx = 0; idx < samples_to_load; idx++) {
        uint32_t i = indices[idx];
        

        uint8_t label = all_labels[i];
        labels.push_back(static_cast<int>(label));

        Matrix img = Matrix::Zero(num_rows * num_cols, 1);
        

        for (uint32_t r = 0; r < num_rows; r++) {
            for (uint32_t c = 0; c < num_cols; c++) {
                uint32_t pixel_idx = i * num_rows * num_cols + r * num_cols + c;
                img(r * num_cols + c, 0) = static_cast<double>(all_images[pixel_idx]) / 255.0;
            }
        }
        
        images.push_back(img);
        

    }
    
    img_file.close();
    lbl_file.close();
    std::cout << "Loaded " << images.size() << " images and " << labels.size() << " labels" << std::endl;
    
    if (!labels.empty()) {
        std::vector<int> label_counts(10, 0);
        for (int label : labels) {
            if (label >= 0 && label < 10) {
                label_counts[label]++;
            }
        }
    }
}

int main()
{

    int max_train_samples = 600;
    int max_test_samples = 100;

    std::vector<Matrix> train_images;
    std::vector<int> train_labels;
    std::vector<Matrix> test_images;
    std::vector<int> test_labels;

    std::string train_images_file = "mnist/train-images.idx3-ubyte";
    std::string train_labels_file = "mnist/train-labels.idx1-ubyte";
    std::string test_images_file = "mnist/t10k-images.idx3-ubyte";
    std::string test_labels_file = "mnist/t10k-labels.idx1-ubyte";

    load_mnist(train_images_file, train_labels_file, train_images, train_labels, max_train_samples);
    
    load_mnist(test_images_file, test_labels_file, test_images, test_labels, max_test_samples);


    std::vector<int> label_counts(10, 0);
    for (int label : train_labels) {
        if (label >= 0 && label < 10) {
            label_counts[label]++;
        }
    }

    std::vector<Matrix> x_train, y_train, x_test, y_test;

    int max_per_class = 60;
    preprocess_data(train_images, train_labels, x_train, y_train, max_per_class);
    preprocess_data(test_images, test_labels, x_test, y_test, max_per_class);

    std::cout << "Training samples: " << x_train.size() << std::endl;
    std::cout << "Test samples: " << x_test.size() << std::endl;


    std::vector<Layer*> network;

    std::cout <<"creating the network\n";

    network.push_back(new Convolutional(1, 28, 28, 3, 8));
    network.push_back(new MaxPooling(8, 26, 26));  
    network.push_back(new ReLU());

    network.push_back(new Dense(8 * 13 * 13, 64));
    network.push_back(new ReLU());
    
    network.push_back(new Dense(64, 10));
    network.push_back(new Softmax());


    train(
        network,
        categorical_cross_entropy,
        categorical_cross_entropy_prime,
        x_train,
        y_train,
        10,    //number of training loops
        0.005  // learning rate
    );

    std::cout << "testing CNN network\n";
    int correct = 0;
    const int num_classes = 10;
    std::vector<int> correct_per_class(num_classes, 0);
    std::vector<int> total_per_class(num_classes, 0);
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    for (size_t i = 0; i < x_test.size(); i++)
    {
        Matrix output = predict(network, x_test[i]);

        int pred_idx = 0;
        double max_val = output(0, 0);
        for (int j = 1; j < output.rows(); j++)
        {
            if (output(j, 0) > max_val)
            {
                max_val = output(j, 0);
                pred_idx = j;
            }
        }

        int true_idx = 0;
        max_val = y_test[i](0, 0);
        for (int j = 1; j < y_test[i].rows(); j++)
        {
            if (y_test[i](j, 0) > max_val)
            {
                max_val = y_test[i](j, 0);
                true_idx = j;
            }
        }

        if (true_idx >= 0 && true_idx < num_classes)
        {
            total_per_class[true_idx]++;
            if (pred_idx == true_idx)
            {
                correct++;
                correct_per_class[true_idx]++;
            }
            confusion_matrix[true_idx][pred_idx]++;
        }

        if (i < 10)
        {
            std::cout << "Sample " << i << ": pred: " << pred_idx << ", true: " << true_idx
                        << ", confidence: " << output(pred_idx, 0) << std::endl;
        }
    }

    double overall_accuracy = static_cast<double>(correct) / x_test.size() * 100;

    std::cout << "\nCNN Test Results:" << std::endl;
    std::cout << "Overall Accuracy: " << overall_accuracy << "%" << std::endl;

    std::cout << "\nClass-wise Accuracy:" << std::endl;
    for (int c = 0; c < num_classes; c++)
    {
        double class_acc = total_per_class[c] > 0 ?
            100.0 * correct_per_class[c] / total_per_class[c] : 0.0;
        std::cout << "  Class " << c << ": " << class_acc
                    << "% (" << correct_per_class[c] << "/" << total_per_class[c] << ")" << std::endl;
    }

    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "  ";
    for (int c = 0; c < num_classes; c++)
    {
        printf("%6d", c);
    }
    std::cout << " <- Predicted" << std::endl;

    for (int r = 0; r < num_classes; r++)
    {
        printf("%2d", r);
        for (int c = 0; c < num_classes; c++)
        {
            printf("%6d",confusion_matrix[r][c]);
        }
        std::cout << std::endl;
    }
    std::cout << "^" << std::endl;
    std::cout << "Actual" << std::endl;

    for (auto layer : network)
    {
        delete layer;
    }
    network.clear();

    std::cout << "\nCreating and training dense network\n";
    network.push_back(new Dense(784, 64));
    network.push_back(new ReLU());
    network.push_back(new Dense(64, 10));
    network.push_back(new Softmax());

    train(
        network,
        categorical_cross_entropy,
        categorical_cross_entropy_prime,
        x_train,
        y_train,
        10,   
        0.005  
    );

    std::cout << "testing dense network\n";
    correct = 0;
    std::fill(correct_per_class.begin(), correct_per_class.end(), 0);
    std::fill(total_per_class.begin(), total_per_class.end(), 0);
    for (auto& row : confusion_matrix) {
        std::fill(row.begin(), row.end(), 0);
    }

    for (size_t i = 0; i < x_test.size(); i++)
    {
        Matrix output = predict(network, x_test[i]);

        int pred_idx = 0;
        double max_val = output(0, 0);
        for (int j = 1; j < output.rows(); j++)
        {
            if (output(j, 0) > max_val)
            {
                max_val = output(j, 0);
                pred_idx = j;
            }
        }

        int true_idx = 0;
        max_val = y_test[i](0, 0);
        for (int j = 1; j < y_test[i].rows(); j++)
        {
            if (y_test[i](j, 0) > max_val)
            {
                max_val = y_test[i](j, 0);
                true_idx = j;
            }
        }

        if (true_idx >= 0 && true_idx < num_classes)
        {
            total_per_class[true_idx]++;
            if (pred_idx == true_idx)
            {
                correct++;
                correct_per_class[true_idx]++;
            }
            confusion_matrix[true_idx][pred_idx]++;
        }

        if (i < 10)
        {
            std::cout << "Sample " << i << ": pred: " << pred_idx << ", true: " << true_idx
                        << ", confidence: " << output(pred_idx, 0) << std::endl;
        }
    }

    overall_accuracy = static_cast<double>(correct) / x_test.size() * 100;

    std::cout << "\nDense Network Test Results:" << std::endl;
    std::cout << "Overall Accuracy: " << overall_accuracy << "%" << std::endl;

    std::cout << "\nClass-wise Accuracy:" << std::endl;
    for (int c = 0; c < num_classes; c++)
    {
        double class_acc = total_per_class[c] > 0 ?
            100.0 * correct_per_class[c] / total_per_class[c] : 0.0;
        std::cout << "  Class " << c << ": " << class_acc
                    << "% (" << correct_per_class[c] << "/" << total_per_class[c] << ")" << std::endl;
    }

    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "  ";
    for (int c = 0; c < num_classes; c++)
    {
        printf("%6d", c);
    }
    std::cout << " <- Predicted" << std::endl;

    for (int r = 0; r < num_classes; r++)
    {
        printf("%2d", r);
        for (int c = 0; c < num_classes; c++)
        {
            printf("%6d",confusion_matrix[r][c]);
            
        }
        std::cout << std::endl;
    }
    std::cout << "^" << std::endl;
    std::cout << "Actual" << std::endl;

    for (auto layer : network)
    {
        delete layer;
    }
    return 0;
}

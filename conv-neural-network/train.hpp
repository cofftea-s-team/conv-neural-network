#pragma once
#include "host/utils.hpp"
#define DEBUG

using std::cout;
using std::endl;
#include "cuda/activations/activation.cuh"
#include "host/matrix.hpp"
#include "cuda/matrix.hpp"
#include "activations.hpp"
#include "utils.hpp"

using namespace base;

template<class _Ty>
class NeuralNetwork {
	private:
	host::matrix<_Ty> _InputW;
	host::matrix<_Ty> _OutputW;
	host::matrix<_Ty> _HiddenW;

	host::matrix<_Ty> _InputB;
	host::matrix<_Ty> _OutputB;
	host::matrix<_Ty> _HiddenB;
public:
	NeuralNetwork(int _InputSize, int _OutputSize, int _HiddenSize) {
		_InputW = host::matrix<_Ty>(_InputSize, _HiddenSize);
		_OutputW = host::matrix<_Ty>(_HiddenSize, _HiddenSize);
		_HiddenW = host::matrix<_Ty>(_HiddenSize, _OutputSize);

		_InputB = host::matrix<_Ty>(_InputSize, 1);
		_OutputB = host::matrix<_Ty>(_HiddenSize, 1);
		_HiddenB = host::matrix<_Ty>(_HiddenSize, 1);

		utils::generate_normal(_InputW);
		utils::generate_normal(_OutputW);
		utils::generate_normal(_HiddenW);

		utils::generate_normal(_InputB);
		utils::generate_normal(_OutputB);
		utils::generate_normal(_HiddenB);
	}
	auto forward(host::matrix<_Ty>& _Input) {
		auto _Z1 = _Input.mul(_InputW);
		_Z1 += _InputB;
		_Z1.activate<sigmoid>();
		auto _Z2 = _Z1.mul(_HiddenW);
		_Z2 += _HiddenB;
		_Z2.activate<sigmoid>();
		auto _Z3 = _Z2.mul(_OutputW);
		_Z3 += _OutputB;
		_Z3.activate<sigmoid>();
		return _Z3;
	}
};

template<class _Ty>
inline host::matrix<_Ty> predict(NeuralNetwork<_Ty>& _NN, host::matrix<_Ty>& _Input) {
	return _NN.forward(_Input);
}

template<class _Ty>
inline NeuralNetwork<_Ty> train(host::matrix<_Ty>& _Input, host::matrix<_Ty>& _Labels) {
	int epochs = 1000;
	double learning_rate = 0.005;
	// moon dataset
	NeuralNetwork<_Ty> _NN(_Input.cols(), _Labels.cols(), 4);

	for (int i = 0; i < epochs; i++) {
		auto _Output = _NN.forward(_Input);
		// binary cross entropy
		auto _OutputLog = _Output;
		for (auto& val : _OutputLog) {
			val = std::log(val);
		}
		auto _OutputLog1 = _Output;
		for (auto& val : _OutputLog1) {
			val = std::log(1 - val);
		}
		auto _1Labels = _Labels;
		for (auto& val : _1Labels) {
			val = 1 - val;
		}
		auto _Loss = _Labels.mul(_OutputLog);
		_Loss += _1Labels.mul(_OutputLog1);
		_Loss *= -1;
		_Ty _LossSum = 0;
		for (auto& val : _Loss) {
			_LossSum += val;
		}
		// derivative
		auto _dLoss = _Output;
		_dLoss -= _Labels;
		_dLoss *= learning_rate;
		// update weights
		_NN._OutputW -= _NN._HiddenW.mul(_dLoss);
		_NN._HiddenW -= _NN._Z1.mul(_dLoss);
		_NN._InputW -= _Input.mul(_dLoss);
		// update biases
		_NN._OutputB -= _dLoss;
		_NN._HiddenB -= _dLoss;
		_NN._InputB -= _dLoss;
		if (i % 100 == 0) {
			cout << "Epoch: " << i << " Loss: " << _LossSum << endl;
		}
	}
	return _NN;
}
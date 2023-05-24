#pragma once

#include "../base/utils.hpp"
#include "config.hpp"
#include "neural_network.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include <string>
#include <string_view>
#include <tuple>
#include <fstream>
#include <filesystem>
#include <exception>
#include "../host/matrix.hpp"
#include "../host/vector.hpp"

namespace stdfs = std::filesystem;

namespace cnn {
	/*
		TODO:
		convert to base64 before writing to file
	*/
	class file {
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;
		using vector = typename config::vector;
		using dual_matrix = typename config::dual_matrix;
		
		inline static stdfs::path working_directory{};

		inline file(std::string_view _View, bool _Override = false)
			: _Override(_Override)
		{
			auto _Dir_path = working_directory / stdfs::path(_View).lexically_relative("/");

			if (_Dir_path.has_extension()) {
				std::cerr << "Given path is not a directory!" << std::endl;
				throw std::exception("Given path is not a directory!");
			}

			if (!stdfs::exists(_Dir_path)) {
				stdfs::create_directory(_Dir_path);
			}

			_Path = std::move(_Dir_path);
		}

		template <class... _TLayers>
		inline void save(neural_network<_TLayers...>& _NN) const {
			if (_Requires_permission_to_override()) {
				if (!_Ask_for_permission()) {
					std::cerr << "Permission denied! Saving cancelled." << std::endl;
					return;
				}
			}
			_Write_to_file(_NN._Layers);
		}

		template <class... _TLayers>
		inline void load(neural_network<_TLayers...>& _NN) const {
			_Read_from_file(_NN._Layers);
		}

	private:
		template <class... _TLayers>
		inline void _Write_to_file(std::tuple<_TLayers...>& _T) const {
			size_t _Counter = 0;
			utils::for_each(_T, [&]<class _TLayer>(const _TLayer& _Layer) {
				constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Write_linear(_Layer, _Counter++);
			});
		}

		inline void _Write_linear(const linear& _Layer, size_t _Layer_id) const {
			auto _File_path = _Path / ("layer" + std::to_string(_Layer_id) + ".cnn");
			std::ofstream _File(_File_path, std::ios::binary);
			if (!_File) {
				std::cerr << "Error opening file!" << std::endl;
				throw std::exception("Error opening file!");
			}
			_Write_matrix(_Layer._Weights, _File);
			_Write_matrix(*reinterpret_cast<const matrix*>(&_Layer._Bias), _File);
		}

		inline void _Write_matrix(const host::matrix<value_type>& _M, std::ofstream& _File) const {
			_File.write(reinterpret_cast<const char*>(_M.data()), _M.size() * sizeof(value_type));
		}

		template <class... _TLayers>
		inline void _Read_from_file(std::tuple<_TLayers...>& _T) const {

			size_t _Counter = 0;
			utils::for_each(_T, [&]<class _TLayer>(_TLayer & _Layer) {
				constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Read_linear(_Layer, _Counter++);
			});
		}

		inline void _Read_linear(linear& _Layer, size_t _Layer_id) const {
			auto _File_path = _Path / ("layer" + std::to_string(_Layer_id) + ".cnn");
			std::ifstream _File(_File_path, std::ios::binary);
			
			if constexpr (decltype(_Layer._Weights)::alloc::is_cuda()) {
				host::matrix<value_type> _W_tmp(_Layer._Weights.shape());
				host::vector<value_type, true> _B_tmp(_Layer._Bias.shape());
				_Read_matrix(_W_tmp, _File);
				_Read_matrix(_B_tmp, _File);
				
				_Layer._Weights = _W_tmp;
				_Layer._Bias = _B_tmp;
			}
			else {
				_Read_matrix(_Layer._Weights, _File);
				_Read_matrix(_Layer._Bias, _File);
			}
		}
		template <class _TMatrix>
		inline void _Read_matrix(_TMatrix& _M, std::ifstream& _File) const {
			_File.read(reinterpret_cast<char*>(_M.data()), _M.size() * sizeof(value_type));
		}

		bool _Requires_permission_to_override() const {
			stdfs::path _First_layer_path = _Path / "layer0.txt";
			return !_Override && stdfs::exists(_First_layer_path);
		}

		bool _Ask_for_permission() const {
			std::cout << "Do you want to override the existing files? (y/n)" << std::endl;
			std::string _Answer;
			std::cin >> _Answer;
			_Answer[0] = std::tolower(_Answer[0]);
			return _Answer == "y";
		}

		bool _Override;
		stdfs::path _Path;
	};
}
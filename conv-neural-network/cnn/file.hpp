#pragma once

#include "../base/utils.hpp"
#include "config.hpp"
#include "neural_network.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include <string>
#include <string_view>
#include <tuple>

namespace cnn {
	
	class file {
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;
		using vector = typename config::vector;
		using dual_matrix = typename config::dual_matrix;
		
		inline file(std::string_view _View) 
			: _Path(_View)
		{ }

		template <class... _TLayers>
		inline void save(neural_network<_TLayers...>& _NN) const {
			_Write_to_file(_NN._Layers);
		}

		template <class... _TLayers>
		inline void load(neural_network<_TLayers...>& _NN) const {
			_Read_from_file(_NN._Layers);
		}

	private:
		template <class... _TLayers>
		inline void _Write_to_file(std::tuple<_TLayers...>& _T) const {
			std::ofstream(_Path);
			utils::for_each(_T, [&]<class _TLayer>(const _TLayer& _Layer) {
				constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Write_linear(_Layer);
				
			});
		}

		inline void _Write_linear(const linear& _Layer) const {
			_Write_matrix(host::matrix<value_type>{ _Layer._Weights });
			_Write_matrix(host::matrix<value_type>{ _Layer._Bias });
		}

		inline void _Write_matrix(const host::matrix<value_type>& _M) const {
			std::ofstream _File(_Path, std::ios::binary | std::ios::app);
			_File.write(reinterpret_cast<const char*>(_M.data()), _M.size() * sizeof(value_type));
		}

		template <class... _TLayers>
		inline void _Read_from_file(std::tuple<_TLayers...>& _T) const {
			std::ifstream _File(_Path, std::ios::binary);

			utils::for_each(_T, [&]<class _TLayer>(_TLayer & _Layer) {
				constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Read_linear(_Layer, _File);
			});
		}

		inline void _Read_linear(linear& _Layer, std::ifstream& _File) const {
			_Read_matrix(_Layer._Weights, _File);
			_Read_matrix(_Layer._Bias, _File);
		}

		template <class _HostMat>
		inline void _Read_matrix(_HostMat& _M, std::ifstream& _File) const {
			_File.read(reinterpret_cast<char*>(_M.data()), _M.size() * sizeof(value_type));
		}

		const std::string _Path;
	};
}
#pragma once
#include "cuda_bf16.h"
#include <cassert>


using bfloat16 = nv_bfloat16;

template <class _Activation_fn>
concept activation_fn_t = requires(double _X) {
	{ _Activation_fn::forward(_X) };
	{ _Activation_fn::backward(_X) };
} || requires (double _X) {
	{ _Activation_fn::forward(_X, _X) };
};

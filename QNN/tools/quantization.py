import torch
import torch.nn as nn


def Quantize_w(k):
	class qfn(torch.autograd.Function):

		@staticmethod
		def forward(self, input):
			if k == 32:
				out = input
			elif k == 1:
				out = torch.sign(input)
			else:
				maxW = torch.max(abs(input))
				# out = input / maxW
				out = torch.clamp(input, -maxW, maxW)  # ensure symmetry
				n = pow(2,k-1)
				out = torch.round(out*n )
				out = torch.clamp(out, -n, n-1) *maxW
				out =  torch.round(out)
			# print(out)	
			return out

		@staticmethod
		def backward(self, grad_output):
			grad_input = grad_output.clone()
			return grad_input

	return qfn().apply

def Quantize_a(k):
	class qfn(torch.autograd.Function):

		@staticmethod
		def forward(self, input):
			if k == 32:
				out = input
			elif k == 1:
				out = torch.sign(input)
			else:
				n = pow(2,k-1)
				out = torch.round(input * n) 
				out = torch.clamp(out, -n, n-1)
			return out

		@staticmethod
		def backward(self, grad_output):
			grad_input = grad_output.clone()
			return grad_input
	return qfn().apply

class QLinear(nn.Linear):

	def __init__(self, abits=32, wbits=32, *kargs, **kwargs):
		super(QLinear, self).__init__(*kargs, **kwargs)
		self.abits=abits
		self.wbits=wbits
		self.quant_a = Quantize_a(k=self.abits)
		self.quant_w = Quantize_w(k=self.wbits)

	def forward(self, input):
		input.data=self.quant_a(input.data)
		if not hasattr(self.weight,'org'):
			self.weight.org=self.weight.data.clone()
		self.weight.data=self.quant_w(self.weight.org)
		out = nn.functional.linear(input, self.weight)
		if not self.bias is None:
			self.bias.org=self.bias.data.clone() #(self.abits*self.wbits)*
			quant_a1 = Quantize_w(k=self.abits)  # returns a function
			bias1 = quant_a1(self.bias.org*5)      # apply the function to the data
			out += bias1.view(1, -1).expand_as(out)
		print(torch.max(out))
		return out

class QConv2d(nn.Conv2d):

	def __init__(self, abits=32, wbits=32, *kargs, **kwargs):
		super(QConv2d, self).__init__(*kargs, **kwargs)
		self.abits=abits
		self.wbits=wbits
		self.quant_a = Quantize_a(k=self.abits)
		self.quant_w = Quantize_w(k=self.wbits)

	def forward(self, input):
		if input.size(1) != 1 and input.size(1) != 3:	# For MNIST (Grey-Level Image) and CIFAR-10/ImageNet (RGB Image)
			input.data=self.quant_a(input.data)
		if not hasattr(self.weight,'org'):
			self.weight.org=self.weight.data.clone()
		self.weight.data=self.quant_w(self.weight.org)

		out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

		if not self.bias is None:
			self.bias.org=self.bias.data.clone()
			out += self.bias.view(1, -1, 1, 1).expand_as(out)

		return out

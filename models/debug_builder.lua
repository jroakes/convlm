	 	
local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:make_net(config, vocab_size, hsm)

	-- input: batch_size * context_size
	-- for example 128 * 32

	local input = nn.Identity()()

	hsm = hsm or false

	-- Projection layer
	local lookup = nn.LookupTable(vocab_size, config.vec_size)
	lookup.weight:uniform(-0.1, 0.1)
	local embeddings = lookup(input)

	-- local embeddings = input

	local HighwayConv = require 'models.HighwayConv'

	-- kernel is an array of kernel sizesch
	local kernels = config.kernels

	local layer1 = {}
	local conv_output_size = 0

	for i = 1, #kernels do
		local conv, conv_output
		local conv_layer, max_time, reshaped_max_time
		conv = cudnn.TemporalConvolution(config.vec_size, config.num_feat_maps, kernels[i])
		-- conv = nn.TemporalConvolutionFB(config.vec_size, config.num_feat_maps, kernels[i])
		
		-- print(config.memsize, kernels[i], config.num_feat_maps)
		-- conv = 	cudnn.SpatialConvolution(1, config.num_feat_maps, config.vec_size, kernels[i])
		conv_output = conv(nn.Reshape(config.memsize, config.vec_size, true)(embeddings))

		conv_layer = nn.Reshape(config.num_feat_maps, config.memsize - kernels[i] + 1, true)(conv_output)

		max_time = nn.TemporalKMaxPooling(config.kmax)(conv_layer)

		reshaped_max_time = nn.Squeeze()(nn.Reshape(config.kmax * (config.memsize - kernels[i] + 1), 1, true)(max_time))

		print(config.kmax * (config.memsize - kernels[i] + 1))

    	conv.weight:uniform(-0.01, 0.01)
		conv.bias:zero()
		-- table.insert(layer1, max_time)
		table.insert(layer1, reshaped_max_time)
		conv_output_size = conv_output_size + config.kmax * (config.memsize - kernels[i] + 1)
	end

	-- Concatenate output features
	local conv_layer_concat

	if #layer1 > 1 then
		conv_layer_concat = nn.JoinTable(2)(layer1)
	else
		conv_layer_concat = layer1[1]
	end

	-- local last_layer = nn.Dropout(config.dropout)(conv_layer_concat)
	local last_layer = conv_layer_concat
	
	-- local conv_output_size = (#layer1) * config.num_feat_maps * config.kmax
	-- print(conv_output_size, #layer1)
	local output, output_size

	

	if config.highway_mlp > 0 then
	-- use highway layers
		local HighwayMLP = require 'models.HighwayMLP'
		local highway = HighwayMLP.mlp(conv_output_size, config.highway_mlp, -2, nn.Tanh())
		last_layer = highway(conv_layer_concat)	
	end

	print("Simple Linear")
	local linear_encoder = nn.Linear(conv_output_size, config.vec_size)
	last_layer = linear_encoder(last_layer)
	linear_encoder.weight:normal():mul(0.05)
	

	-- output = last_layer

	local dropout = nn.Dropout(config.dropout)(last_layer)


	if hsm == false then
		local linear = nn.Linear( config.vec_size, vocab_size)
		output_size = vocab_size
		linear.weight:normal():mul(0.05)
		linear.bias:zero()
		local softmax = cudnn.LogSoftMax()

		output = softmax(linear(dropout))

	else
		output = nn.Identity()(dropout)
		output_size = (#layer1) * config.num_feat_maps
	end

	local model = nn.gModule({input}, {output})

	model:cuda()


	return model, output_size


end

return ModelBuilder
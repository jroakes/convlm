	 	
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

	-- kernel is an array of kernel sizesch
	local kernels = config.kernels

	local layer1 = {}

	for i = 1, #kernels do
		local conv, conv_output
		local conv_layer, max_time
		conv = cudnn.TemporalConvolution(config.vec_size, config.num_feat_maps, kernels[i])
		-- conv = nn.TemporalConvolutionFB(config.vec_size, config.num_feat_maps, kernels[i])
		
		
		-- print(config.memsize, kernels[i], config.num_feat_maps)
		-- conv = 	cudnn.SpatialConvolution(1, config.num_feat_maps, config.vec_size, kernels[i])
		conv_output = conv(nn.Reshape(config.memsize, config.vec_size, true)(embeddings))

		conv_layer = nn.Reshape(config.num_feat_maps, config.memsize - kernels[i] + 1, true)(conv_output)

		-- max_time = nn.Mean(3)(cudnn.Tanh()(conv_layer))	

		max_time = nn.Max(3)(cudnn.Tanh()(conv_layer))			

    	conv.weight:uniform(-0.01, 0.01)
		conv.bias:zero()
		table.insert(layer1, max_time)
	end

	-- Concatenate output features
	local conv_layer_concat

	if #layer1 > 1 then
		conv_layer_concat = nn.JoinTable(2)(layer1)
	else
		conv_layer_concat = layer1[1]
	end

	local last_layer = conv_layer_concat
	
	local output, output_size

	if config.highway_mlp > 0 then
	-- use highway layers
		local HighwayMLP = require 'models.HighwayMLP'
		local highway = HighwayMLP.mlp((#layer1) * config.num_feat_maps, config.highway_mlp, -2, nn.Tanh())
		last_layer = highway(conv_layer_concat)
	end

	local dropout = nn.Dropout(config.dropout)(last_layer)


	if hsm == false then
		local linear = nn.Linear( (#layer1) * config.num_feat_maps, vocab_size)
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
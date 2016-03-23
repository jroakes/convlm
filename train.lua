-- This file trains and tests the RNN from a batch loader.
require('torch')
require('nn')
require('nnx')
require('nngraph')
require('options')
-- require 'utils.misc'

require('utils.batchloader')
require('utils.textsource')
model_utils = require('utils/model_utils')
require('models.builder')

require 'cutorch'
require 'cunn'
require 'cunnx'
require 'cudnn'
require 'fbcunn'
cutorch.setDevice(1)
cutorch.manualSeed(1990)

-- Parse arguments
local cmd = RNNOption()
g_params = cmd:parse(arg)
torch.manualSeed(1990)

cmd:print_params(g_params)

-- build the torch dataset
local g_dataset = TextSource(g_params.dataset)
local vocab_size = g_dataset:get_vocab_size()

local loader = BatchLoader(g_params.dataset, g_dataset)
local criterion
local mapping
-- Preparing for the hierarchical softmax
hsm = false
smt = false
if string.find(g_params.model.name, '_hsm') then
	-- Using facebook's hierarchical softmax (2 layers)
	hsm = true
	g_dataset:create_frequency_clusters(g_params.dataset)
	mapping = g_dataset.dict.mapping
	-- g_dataset:create_clusters(g_params.dataset)
	-- mapping = g_dataset.dict.clusters
elseif string.find(g_params.model.name, '_smt') then
	smt = true
	-- Using softmax tree (deep hierarchical softmax)
	g_dataset:create_frequency_tree(g_params.dataset)

else
	criterion = nn.ClassNLLCriterion(nil, false)
end

local g_dictionary = g_dataset.dict


local hierarchical_output = false

if hsm == true or smt == true then
	hierarchical_output = true
end

-- Put a lookuptable separately to use its sparse update (much faster)
-- local lookuptable = nn.LookupTable(vocab_size, g_params.model.vec_size)
-- lookuptable.weight:uniform(-0.1, 0.1)
-- lookuptable:cuda()

local model, output_size = ModelBuilder:make_net(g_params.model, vocab_size, hierarchical_output)
-- params, grad_params = model_utils.combine_all_parameters(model)
params, grad_params = model:getParameters()

-- params:uniform(-0.05, 0.05)

-- Initialise the hierarchical softmax
if hsm == true then
	-- criterion = nn.HLogSoftMax(mapping, output_size)
	-- hsm_params, hsm_grad_params = criterion:getParameters()
	-- hsm_params:uniform(-0.05, 0.05)
	criterion = nn.HSM(mapping, output_size)
	criterion:reset(0. , 0.)
end

-- Initialise the softmax tree
local softmaxtree
if smt == true then
	require 'models.NLLTreeCriterion'
	softmaxtree = nn.SoftMaxTree(output_size, g_dictionary.tree, g_dictionary.root_id)
	softmaxtree:zeroGradParameters()
	softmaxtree:reset(-0.1, 0.1)
	softmaxtree:cuda()
	criterion = nn.NLLTreeCriterion(false) -- doesn't average loss over batch
	criterion:cuda()
end


local total_params = params:nElement()

if hsm == true then
	local hsm_params = criterion:getParameters()
	total_params = total_params + hsm_params:nElement()
end	

print("Total parameters of model: " .. total_params)


criterion:cuda()

function eval(split)

	model:evaluate()
	loader:reset_batch_pointer(split)

	local n_batches = loader.split_sizes[split]
	local total_loss = 0
	local total_samples = 0

	for i = 1, n_batches do
		xlua.progress(i, n_batches)

		
		local context, target = loader:next_batch(split)
		local batch_size = target:size(1)

		-- local embeddings = lookuptable:forward(context)

		local net_output = model:forward(context)

		-- Get loss through the decoder
		local loss, tree_output
		if smt == false then
			-- Same for HSM and SM
			loss = criterion:forward(net_output, target)
		else
			tree_output = softmaxtree:forward{net_output, target}
			loss = criterion:forward(tree_output, target)
		end
		

		if type(loss) ~= 'number' then
            loss = loss[1]
        end

		total_loss = total_loss + loss
		total_samples = total_samples + batch_size

	end

	total_loss = total_loss / total_samples

	local perplexity = torch.exp(total_loss)

	return perplexity

end

function train_epoch(learning_rate, gradient_clip)

	model:training()
	loader:reset_batch_pointer(1)

	
	-- if hsm == true then hsm_grad_params:zero() end
	local speed
	local n_batches = loader.split_sizes[1]
	local total_loss = 0
	local total_samples = 0

	local timer = torch.tic()

	for i = 1, n_batches do
		-- grad_params:zero()
		model:zeroGradParameters()
		xlua.progress(i, n_batches)

		-- forward pass 
		local context, target = loader:next_batch(split)
		local batch_size = target:size(1)


		-- local embeddings = lookuptable:forward(context)
		local net_output = model:forward(context)
		-- print(net_output:size())

		local loss, tree_output
		if smt == false then
			-- Same for HSM and SM
			loss = criterion:forward(net_output, target)
		else
			tree_output = softmaxtree:forward{net_output, target}
			loss = criterion:forward(tree_output, target)
		end

		
		if type(loss) ~= 'number' then
            loss = loss[1]
        end

		total_loss = total_loss + loss
		total_samples = total_samples + batch_size

		-- backward pass
		local dloss, treeloss

        if hsm == true then
        	dloss = criterion:updateGradInput(net_output, target)
        	criterion:accGradParameters(net_output, target, -learning_rate, true)
        	criterion.class_grad_bias:zero()
        	criterion.cluster_grad_bias:zero()
        elseif smt == true then
        	treeloss = criterion:backward(tree_output, target)
        	dloss = softmaxtree:backward({net_output, target}, treeloss)[1]

        	-- Update softmaxtree params
        	softmaxtree:updateParameters(learning_rate)
        	softmaxtree:zeroGradParameters()
        else
        	dloss = criterion:backward(net_output, target)
        end

  		model:backward(context, dloss)

		-- Control if gradient too big
		local norm = grad_params:norm()

		if norm > gradient_clip then
            grad_params:mul(gradient_clip / norm)
        end

		-- params:add(grad_params:mul(-learning_rate))
		model:updateParameters(learning_rate)

	end

	local elapse = torch.toc(timer)
	local speed = math.floor(total_samples / elapse)

	total_loss = total_loss / total_samples

	local perplexity = torch.exp(total_loss)

	return perplexity, speed
end





local function run(n_epochs)
	
	local val_loss = {}
	local pp = eval(2)
	print(pp)
	val_loss[0] = pp

	local learning_rate = g_params.trainer.initial_learning_rate
	local gradient_clip = g_params.trainer.gradient_clip

	for epoch = 1, n_epochs do
		
		local train_loss, wps = train_epoch(learning_rate, gradient_clip)

		val_loss[epoch] = eval(2)

		if val_loss[epoch] >= val_loss[epoch - 1] * 0.9999 then
			learning_rate = learning_rate / g_params.trainer.learning_rate_shrink
		end


		local stat = {perplexity = train_loss , epoch = epoch,
                valid_perplexity = val_loss[epoch], LR = learning_rate, speed = wps}

        print(stat)

	end

	print(eval(3))
	
end

run(g_params.trainer.n_epochs)
-- print(output)

-- print(context, target)

-- test model ...

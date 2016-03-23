------------------------------------------------------------------------
--[[ NLLTreeCriterion ]]--
-- Negative Log Likelihood for SoftMaxTrees.
-- Used for maximizing the likelihood of SoftMaxTree outputs.
-- SoftMaxTree outputs a column tensor representing the log likelihood
-- of each target in the batch. Thus SoftMaxTree requires the targets.
-- So this Criterion only computes the negative of those outputs, as 
-- well as its corresponding gradients.
------------------------------------------------------------------------
local NLLTreeCriterion, parent = torch.class("nn.NLLTreeCriterion", "nn.Criterion")

function NLLTreeCriterion:__init(sizeAverage)
   
   if sizeAverage == nil then
   	sizeAverage = true
   end	
   self.sizeAverage = sizeAverage

   if self.sizeAverage == true then
   	self._module = nn.Mean()
   else
      self._module = nn.Sum()
   end
   parent.__init(self)
   self._output_grad = torch.Tensor{-1}
end

function NLLTreeCriterion:updateOutput(input, target)
   return -self._module:forward(input)[1]
end

function NLLTreeCriterion:updateGradInput(input, target)
   return self._module:backward(input, self._output_grad)
end

### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 3386da1c-34e9-4d8c-9f2c-2a2a7a06a9da
begin
	using Pkg
	using CSV
	using DataFrames
	using PlutoUI
	using Statistics
	using RDatasets
	using StatsBase
	using Plots
	using EvalMetrics
end

# ╔═╡ f472c687-c9aa-4e96-8bb4-69487b065688
Pkg.activate()

# ╔═╡ 54823462-d3a4-11eb-2ae9-b9480ca5e0ea
md" # Logistic Regression with Julia"

# ╔═╡ 0a1d114d-6b87-402b-8ac1-702e2b81784e
path = pwd()

# ╔═╡ 84553e4a-caad-4bb9-89ac-adf9c473991c
md" ## Imports"

# ╔═╡ 15d1963a-e707-434b-9de2-2340d8d931dd
md" ## Load the data"

# ╔═╡ c1a9634e-b377-4de4-8a5d-76f042d7a661
dataset = CSV.read(path*"/data/heart.csv", DataFrame)

# ╔═╡ 73d1535f-298e-44a6-8bfc-bb6e9caa2d6e
md" ### Dataset properties"

# ╔═╡ 4b242413-57ad-4a3d-8580-8b6e95e09729
size(dataset)

# ╔═╡ dcbcec0f-0289-4d5f-97a5-109eea6cdff4
md" #### Column Names"

# ╔═╡ bdcd4061-af34-402f-96fd-2376c78eeafb
with_terminal() do

	for cols in names(dataset)
		println(cols)
	end
end

# ╔═╡ 7d8b58c8-63a4-4ff9-80a3-d24d958e8d97
md" #### Stats"

# ╔═╡ 1dda4976-c9a4-4fba-a6f1-c1cd2358bcda
describe(dataset)

# ╔═╡ faffea49-8075-44fd-a5e5-22b08567c0e8
md" #### Backup of the dataset"

# ╔═╡ 4307d296-a946-47a6-8e94-6c83aa16a8d8
backup = deepcopy(dataset)

# ╔═╡ 723d33d9-eb23-4252-bcff-03a86451e0b6
md" #### Create matrices"

# ╔═╡ 1225d3d0-d676-41de-84fb-4ddff30593f6
X = dataset[:,1:13]

# ╔═╡ 19529f9e-60e5-4563-9f8a-0e31e224813e
xMatrix = Matrix(X)

# ╔═╡ c070a6fa-10e4-4fdf-a39b-c73857848ee6
Y = dataset[:,14]

# ╔═╡ 07cb3918-1573-47c2-9f40-44d16f26b26c
yVector = Vector(Y)

# ╔═╡ d1001996-f47b-48a9-8344-79ef8b2934f2
trainSize = 0.7

# ╔═╡ 2d1750d2-4341-410d-9671-5735edb0db3d
dataSize = size(dataset)[1]

# ╔═╡ 00d9caa8-9c66-46ae-a44c-8a235280a0d1
trainIndex = trunc(Int, trainSize * dataSize)

# ╔═╡ 23bad6b3-a37c-47a6-a1dd-0c2a96ba484e
xTrain = xMatrix[1:trainIndex,:]

# ╔═╡ cc512677-02b5-4b44-9698-7e56dd0305aa
xTest = xMatrix[1+trainIndex:end,:]

# ╔═╡ e5cac00b-21ad-4ca9-b4ad-4c3af5993586
yTrain = yVector[1:trainIndex,:]

# ╔═╡ ccad36d3-7412-4276-82d5-d59c9d9ee7e7
yTest = yVector[1+trainIndex:end,:]

# ╔═╡ 10b3c10a-a8ca-4ebb-9fa8-d72ddb82a518
function scalingParameters(initFeatures)
	featureMean = mean(initFeatures,dims=1)
	featureDev = std(initFeatures,dims=1)
	
	return (featureMean, featureDev)
	
end

# ╔═╡ c2ea9a76-c0e3-4300-9267-6d76373d1e63
scaleParams = scalingParameters(xMatrix)

# ╔═╡ 6b6d9a26-f022-415b-a2c8-edefa51860da
function scaleFeatures(featureMatrix, sParams)
	scaledFeatures =(featureMatrix .- sParams[1]) ./ sParams[2]	
end

# ╔═╡ e89ab3d3-d392-4f2b-a75c-9274325f3483
scaledTrainFeatures = scaleFeatures(xTrain, scaleParams)

# ╔═╡ 79743dee-e7d2-4155-935a-031e019291c6
scaledTestFeatures = scaleFeatures(xTest,scaleParams)

# ╔═╡ f7fef911-41d4-435e-b839-5a59a9b51968
function sigmoid(z)
	1 ./(1 .+ exp.(.-z))
end

# ╔═╡ e1a385af-97c9-400d-bb99-58a50d798c5e
function get_cost(aug_features, outcome, weights, reg_param)
sample_count = length(outcome)
hypothesis = sigmoid(aug_features * weights)
cost_part_1 = ((-outcome)' * log.(hypothesis))[1]
cost_part_2 = ((1 .- outcome)' * log.(1 .- hypothesis))[1]
lambda_regul = (reg_param/(2 * sample_count) * sum(weights[2:end] .^ 2))
error = (1/sample_count) * (cost_part_1 - cost_part_2) + lambda_regul
grad_error_all = ((1/sample_count) * (aug_features') * (hypothesis - outcome)) +
((1/sample_count) * (reg_param * weights))
grad_error_all[1] = ((1/sample_count) * (aug_features[:,1])' * (hypothesis -
outcome))[1]
return(error, grad_error_all)
end

# ╔═╡ d4a85aff-7449-40da-ba66-50b93e8ff2a3
function train_model(features, outcome, reg_param, alpha, max_iter)
sample_count = length(outcome)
aug_features = hcat(ones(sample_count,1), features)
feature_count = size(aug_features)[2]
weights = zeros(feature_count)
errors = zeros(max_iter)
for i in 1:max_iter
error_and_grad_tp = get_cost(aug_features, outcome, weights, reg_param)
errors[i] = error_and_grad_tp[1]
weights = weights - (alpha * error_and_grad_tp[2])
end
return (errors, weights)
end

# ╔═╡ 69275bf6-61dd-4a54-824e-eae4d0686794
trainWeightErrors = train_model(scaledTrainFeatures, yTrain,0.001,0.2,4000)

# ╔═╡ b884a701-c763-4421-9f33-c1ba24338798
plot(trainWeightErrors[1],label="Cost",ylabel="Cost",xlabel="Number of Iteration",
title="Cost Per Iteration")

# ╔═╡ 819ebbbb-c54a-4be2-b0f4-411c953fede0
function get_predictions(features, weights)
total_entry_count = size(features)[1]
aug_features = hcat(ones(total_entry_count, 1), features)
preds = sigmoid(aug_features * weights)
return
end

# ╔═╡ 0e3cf26c-7ccd-49aa-b12c-7ef5d94aaa5b
function get_predicted_classes(preds, threshold)
return preds .>= threshold
end

# ╔═╡ c7e6fb2b-965c-42e9-aa49-15738dfe582c
tr_predicted_cls = get_predicted_classes(get_predictions(scaledTrainFeatures,
trainWeightErrors[2]), 0.5)

# ╔═╡ 3ccbf6e1-8433-47c2-8ef3-905b62a154b2
ts_predicted_cls=get_predicted_classes(get_predictions(scaledTestFeatures,trainWeightErrors[2]), 0.5)

# ╔═╡ a34d3980-bd05-44e4-8194-a83cb873259d
train_conf_matrix = ConfusionMatrix(yTrain, tr_predicted_cls)

# ╔═╡ 1d528d36-d365-4239-b9d1-e9ac978e84d3
test_conf_matrix = ConfusionMatrix(189, 1083, 14, 1075, 8, 175)

# ╔═╡ 45fbafa2-8938-4db6-866e-0a1c8e6fdcd8
accuracy(test_conf_matrix)

# ╔═╡ 6728b063-9aa8-40da-b4cb-e92d5cd51bf6
recall(test_conf_matrix)

# ╔═╡ 9e13fb3d-f50f-4d28-8199-541e8079bdaf
precision(test_conf_matrix)

# ╔═╡ Cell order:
# ╟─54823462-d3a4-11eb-2ae9-b9480ca5e0ea
# ╠═0a1d114d-6b87-402b-8ac1-702e2b81784e
# ╟─84553e4a-caad-4bb9-89ac-adf9c473991c
# ╠═3386da1c-34e9-4d8c-9f2c-2a2a7a06a9da
# ╠═f472c687-c9aa-4e96-8bb4-69487b065688
# ╟─15d1963a-e707-434b-9de2-2340d8d931dd
# ╠═c1a9634e-b377-4de4-8a5d-76f042d7a661
# ╟─73d1535f-298e-44a6-8bfc-bb6e9caa2d6e
# ╠═4b242413-57ad-4a3d-8580-8b6e95e09729
# ╟─dcbcec0f-0289-4d5f-97a5-109eea6cdff4
# ╠═bdcd4061-af34-402f-96fd-2376c78eeafb
# ╟─7d8b58c8-63a4-4ff9-80a3-d24d958e8d97
# ╠═1dda4976-c9a4-4fba-a6f1-c1cd2358bcda
# ╟─faffea49-8075-44fd-a5e5-22b08567c0e8
# ╠═4307d296-a946-47a6-8e94-6c83aa16a8d8
# ╟─723d33d9-eb23-4252-bcff-03a86451e0b6
# ╠═1225d3d0-d676-41de-84fb-4ddff30593f6
# ╠═19529f9e-60e5-4563-9f8a-0e31e224813e
# ╠═c070a6fa-10e4-4fdf-a39b-c73857848ee6
# ╠═07cb3918-1573-47c2-9f40-44d16f26b26c
# ╠═d1001996-f47b-48a9-8344-79ef8b2934f2
# ╠═2d1750d2-4341-410d-9671-5735edb0db3d
# ╠═00d9caa8-9c66-46ae-a44c-8a235280a0d1
# ╠═23bad6b3-a37c-47a6-a1dd-0c2a96ba484e
# ╠═cc512677-02b5-4b44-9698-7e56dd0305aa
# ╠═e5cac00b-21ad-4ca9-b4ad-4c3af5993586
# ╠═ccad36d3-7412-4276-82d5-d59c9d9ee7e7
# ╠═10b3c10a-a8ca-4ebb-9fa8-d72ddb82a518
# ╠═c2ea9a76-c0e3-4300-9267-6d76373d1e63
# ╠═6b6d9a26-f022-415b-a2c8-edefa51860da
# ╠═e89ab3d3-d392-4f2b-a75c-9274325f3483
# ╠═79743dee-e7d2-4155-935a-031e019291c6
# ╠═f7fef911-41d4-435e-b839-5a59a9b51968
# ╠═e1a385af-97c9-400d-bb99-58a50d798c5e
# ╠═d4a85aff-7449-40da-ba66-50b93e8ff2a3
# ╠═69275bf6-61dd-4a54-824e-eae4d0686794
# ╠═b884a701-c763-4421-9f33-c1ba24338798
# ╠═819ebbbb-c54a-4be2-b0f4-411c953fede0
# ╠═0e3cf26c-7ccd-49aa-b12c-7ef5d94aaa5b
# ╠═c7e6fb2b-965c-42e9-aa49-15738dfe582c
# ╠═3ccbf6e1-8433-47c2-8ef3-905b62a154b2
# ╠═a34d3980-bd05-44e4-8194-a83cb873259d
# ╠═1d528d36-d365-4239-b9d1-e9ac978e84d3
# ╠═45fbafa2-8938-4db6-866e-0a1c8e6fdcd8
# ╠═6728b063-9aa8-40da-b4cb-e92d5cd51bf6
# ╠═9e13fb3d-f50f-4d28-8199-541e8079bdaf

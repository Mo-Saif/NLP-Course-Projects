using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging

# ATYPE = Array{Float32}
ATYPE = KnetArray{Float32}

const datadir = "nn4nlp-code/data/ptb"
isdir(datadir) || run(`git clone https://github.com/neubig/nn4nlp-code.git`)

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    word_count = Dict{String,Int}()
    w2i = Dict{String,Int}()
    i2w = Vector{String}()
    int_unk = get!(w2i, unk, 1+length(w2i))
    int_eos = get!(w2i, eos, 1+length(w2i))
    for line in eachline(file)
        line = tokenizer(line)
        for word in line
            if haskey(word_count, word)
                word_count[word] += 1
            else
                word_count[word] = 1
            end
        end
    end
    word_count = collect(word_count)
    sort!(word_count, rev=true, by=x->x[2])
    # constructing w2i
    for pair in word_count
        if pair[2] >= mincount
            get!(w2i, pair[1], 1+length(w2i))
            if length(w2i) >= vocabsize
                break
            end
        end
    end
    w2i_array = collect(w2i)
    sort!(w2i_array, by=x->x[2])
    for pair in w2i_array
        push!(i2w, pair[1])
    end
    return Vocab(w2i, i2w, int_unk, int_eos, tokenizer)
end

@info "Testing Vocab"
f = "$datadir/train.txt"
v = Vocab(f)
@test all(v.w2i[w] == i for (i,w) in enumerate(v.i2w))
@test length(Vocab(f).i2w) == 10000
@test length(Vocab(f, vocabsize=1234).i2w) == 1234
@test length(Vocab(f, mincount=5).i2w) == 9859

train_vocab = Vocab("$datadir/train.txt")

struct TextReader
    file::String
    vocab::Vocab
end

function Base.iterate(r::TextReader, s=nothing)
    s == nothing && (s = open(r.file))
    if eof(s)
        close(s)
        return nothing
    end
    line = readline(s)
    line = r.vocab.tokenizer(line)
    line_inds = Int[]
    for word in line
        push!(line_inds, get(r.vocab.w2i, word, r.vocab.unk))
    end
    return line_inds, s
end

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

@info "Testing TextReader"
train_sentences, valid_sentences, test_sentences =
    (TextReader("$datadir/$file.txt", train_vocab) for file in ("train","valid","test"))
@test length(first(train_sentences)) == 24
@test length(collect(train_sentences)) == 42068
@test length(collect(valid_sentences)) == 3370
@test length(collect(test_sentences)) == 3761

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    return Embed(param(embedsize, vocabsize, atype = ATYPE))
end

function (l::Embed)(x)
    l.w[:,x]
end

@info "Testing Embed"
Knet.seed!(1)
embed = Embed(100,10)
input = rand(1:100, 2, 3)
output = embed(input)
@test size(output) == (10, 2, 3)
@test norm(output) ≈ 0.59804f0

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    return Linear(param(outputsize, inputsize, atype = ATYPE), param0(outputsize, atype = ATYPE))
end

function (l::Linear)(x)
    l.w * x .+ l.b
end

@info "Testing Linear"
Knet.seed!(1)
linear = Linear(100,10)
input = oftype(linear.w, randn(Float32, 100, 5))
output = linear(input)
@test size(output) == (10, 5)
@test norm(output) ≈ 5.5301356f0

struct NNLM; vocab; windowsize; embed; hidden; output; dropout; end

function NNLM(vocab::Vocab, windowsize::Int, embedsize::Int, hiddensize::Int, dropout::Real)
    embed = Embed(length(vocab.i2w), embedsize)
    hidden = Linear(windowsize * embedsize, hiddensize)
    output = Linear(hiddensize, length(vocab.i2w))
    return NNLM(vocab, windowsize, embed, hidden, output, dropout)
end

# Default model parameters
HIST = 3
EMBED = 128
HIDDEN = 128
DROPOUT = 0.5
VOCAB = length(train_vocab.i2w)

@info "Testing NNLM"
model = NNLM(train_vocab, HIST, EMBED, HIDDEN, DROPOUT)
@test model.vocab === train_vocab
@test model.windowsize === HIST
@test size(model.embed.w) == (EMBED,VOCAB)
@test size(model.hidden.w) == (HIDDEN,HIST*EMBED)
@test size(model.hidden.b) == (HIDDEN,)
@test size(model.output.w) == (VOCAB,HIDDEN)
@test size(model.output.b) == (VOCAB,)
@test model.dropout == 0.5

function pred_v1(m::NNLM, hist::AbstractVector{Int})
    @assert length(hist) == m.windowsize
    out = dropout(reshape(m.embed(hist), (:)), m.dropout)
    out = dropout(tanh.(m.hidden(out)), m.dropout)
    out = m.output(out)
end

@info "Testing pred_v1"
h = repeat([model.vocab.eos], model.windowsize)
p = pred_v1(model, h)
@test size(p) == size(train_vocab.i2w)


# This predicts the scores for the whole sentence, will be used for later testing.
function scores_v1(model, sent)
    hist = repeat([ model.vocab.eos ], model.windowsize)
    scores = []
    for word in [ sent; model.vocab.eos ]
        push!(scores, pred_v1(model, hist))
        hist = [ hist[2:end]; word ]
    end
    hcat(scores...)
end

sent = first(train_sentences)
@test size(scores_v1(model, sent)) == (length(train_vocab.i2w), length(sent)+1)

function generate(m::NNLM; maxlength=30)
    sent = []
    hist = repeat([m.vocab.eos], m.windowsize)
    for i in 1:maxlength
        probs = softmax(pred_v1(m, hist))
        word = sample(m.vocab.i2w, ProbabilityWeights(Array(probs)))
        word_ind = m.vocab.w2i[word]
        push!(sent, word)
        push!(hist, word_ind); hist = hist[2:end]
        if word_ind == m.vocab.eos
            break
        end
    end
    return join(sent, " ")
end

@info "Testing generate"
s = generate(model, maxlength=5)
@test s isa String
@test length(split(s)) <= 5

function loss_v1(m::NNLM, sent::AbstractVector{Int}; average = true)
    loss = 0
    count = 0
    hist = repeat([m.vocab.eos], m.windowsize)
    for target_word in vcat(sent, m.vocab.eos)
        scores = pred_v1(m, hist)
        loss += nll(scores, [target_word])
        push!(hist, target_word); hist = hist[2:end]
        count += 1
    end
    if average == true
        return loss/count
    else
        return loss, count
    end
end

@info "Testing loss_v1"
s = first(train_sentences)
avgloss = loss_v1(model,s)
(tot, cnt) = loss_v1(model, s, average = false)
@test 9 < avgloss < 10
@test cnt == length(s) + 1
@test tot/cnt ≈ avgloss

function maploss(lossfn, model, data; average = true)
    total_loss = 0
    total_num_words = 0
    for sent in data
        loss, num_words = lossfn(model, sent, average=false)
        total_loss += loss; total_num_words += num_words
    end
    if average == true
        return total_loss/total_num_words
    else
        return total_loss, total_num_words
    end
end

@info "Testing maploss"
tst100 = collect(take(test_sentences, 100))
avgloss = maploss(loss_v1, model, tst100)
@test 9 < avgloss < 10
(tot, cnt) = maploss(loss_v1, model, tst100, average = false)
@test cnt == length(tst100) + sum(length.(tst100))
@test tot/cnt ≈ avgloss

@info "Timing loss_v1 with 1000 sentences"
tst1000 = collect(take(test_sentences, 1000))
@time maploss(loss_v1, model, tst1000)

@info "Timing loss_v1 training with 100 sentences"
trn100 = ((model,x) for x in collect(take(train_sentences, 100)))
@time sgd!(loss_v1, trn100)

function pred_v2(m::NNLM, hist::AbstractMatrix{Int})
    out = m.embed(hist)
    out = reshape(out, (:, size(out,3)))
    out = dropout(out, m.dropout)
    out = dropout(tanh.(m.hidden(out)), m.dropout)
    out = m.output(out)
end

@info "Testing pred_v2"

function scores_v2(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    return pred_v2(model, hist)
end

sent = first(test_sentences)
s1, s2 = scores_v1(model, sent), scores_v2(model, sent)
@test size(s1) == size(s2) == (length(train_vocab.i2w), length(sent)+1)
@test s1 ≈ s2

function loss_v2(m::NNLM, sent::AbstractVector{Int}; average = true)
    hist = [repeat([m.vocab.eos], m.windowsize); sent]
    hist = vcat((hist[i : end+i-m.windowsize]' for i in 1 : m.windowsize)...)
    scores = pred_v2(m, hist) 
    return nll(scores, vcat(sent, [m.vocab.eos]), average = average)
end

@info "Testing loss_v2"
s = first(test_sentences)
@test loss_v1(model, s) ≈ loss_v2(model, s)
tst100 = collect(take(test_sentences, 100))
@test maploss(loss_v1, model, tst100) ≈ maploss(loss_v2, model, tst100)

@info "Timing loss_v2  with 10K sentences"
tst10k = collect(take(train_sentences, 10000))
@time maploss(loss_v2, model, tst10k)

@info "Timing loss_v2 training with 1000 sentences"
trn1k = ((model,x) for x in collect(take(train_sentences, 1000)))
@time sgd!(loss_v2, trn1k)

function pred_v3(m::NNLM, hist::Array{Int})
    out = m.embed(hist)
    E, N, B, S = size(out)
    out = reshape(out, (E*N, B*S))
    out = dropout(out, m.dropout)
    out = dropout(tanh.(m.hidden(out)), m.dropout)
    out = reshape(m.output(out), (length(m.vocab.i2w), B, S))
end

@info "Testing pred_v3"

function scores_v3(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    hist = reshape(hist, size(hist,1), 1, size(hist,2))
    return pred_v3(model, hist)
end

sent = first(train_sentences)
@test scores_v2(model, sent) ≈ scores_v3(model, sent)[:,1,:]

function mask!(a,pad)
    num_rows, num_cols = size(a)
    for row in 1:num_rows
        for column in num_cols:-1:1
            if a[row, column] != pad || a[row, column-1] != pad
                break
            else
                a[row, column] = 0
            end
        end
    end
    return a
end

@info "Testing mask!"
a = [1 2 1 1 1; 2 2 2 1 1; 1 1 2 2 2; 1 1 2 2 1]
@test mask!(a,1) == [1 2 1 0 0; 2 2 2 1 0; 1 1 2 2 2; 1 1 2 2 1]

function loss_v3(m::NNLM, batch::AbstractMatrix{Int}; average = true)
    # batch : BxS
    # hist : N×B×S
    V, N, B = length(m.vocab.i2w), m.windowsize, size(batch,1)
    batch2 = batch[:, 1:end-1]
    batch2 = hcat(repeat([m.vocab.eos], B, N), batch2)
    S = size(batch2,2)
    hist = []
    for i in 1:B
        sent = batch2[i,:]
        sent = vcat((sent[i:end+i-m.windowsize]' for i in 1:m.windowsize)...)
        push!(hist, sent)
    end
    hist = hcat(hist...)
    hist = reshape(hist, (N, S-2, B))
    hist = permutedims(hist, (1,3,2))
#     @show hist
    scores = pred_v3(m, hist)
    answers = copy(batch) 
    mask!(answers, m.vocab.eos)
#     @show size(scores)
#     @show size(answers)
    return nll(scores, answers, average = average)
end

@info "Testing loss_v3"
s = first(test_sentences)
b = [ s; model.vocab.eos ]'
@test loss_v2(model, s) ≈ loss_v3(model, b)

struct LMData
    src::TextReader
    batchsize::Int
    maxlength::Int
    bucketwidth::Int
    buckets
end

function LMData(src::TextReader; batchsize = 64, maxlength = typemax(Int), bucketwidth = 10)
    numbuckets = min(128, maxlength ÷ bucketwidth)
    buckets = [ [] for i in 1:numbuckets ]
    LMData(src, batchsize, maxlength, bucketwidth, buckets)
end

Base.IteratorSize(::Type{LMData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{LMData}) = Base.HasEltype()
Base.eltype(::Type{LMData}) = Matrix{Int}

function Base.iterate(d::LMData, state=nothing)
    if state == nothing
        for b in d.buckets; empty!(b); end
    end
    bucket,ibucket = nothing,nothing
    while true
        iter = (state === nothing ? iterate(d.src) : iterate(d.src, state))
        if iter === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent, state = iter
            if length(sent) > d.maxlength || length(sent) == 0; continue; end
            ibucket = min(1 + (length(sent)-1) ÷ d.bucketwidth, length(d.buckets))
            bucket = d.buckets[ibucket]
            push!(bucket, sent)
            if length(bucket) === d.batchsize; break; end
        end
    end
    if bucket === nothing; return nothing; end
    batchsize = length(bucket)
    maxlen = maximum(length.(bucket))
    batch = fill(d.src.vocab.eos, batchsize, maxlen + 1)
    for i in 1:batchsize
        batch[i, 1:length(bucket[i])] = bucket[i]
    end
    empty!(bucket)
    return batch, state
end

@info "Timing loss_v2 and loss_v3 at various batch sizes"
@info loss_v2; test_collect = collect(test_sentences)
GC.gc(); @time p2 = maploss(loss_v2, model, test_collect)
for B in (1, 8, 16, 32, 64, 128, 256)
    @info loss_v3,B; test_batches = collect(LMData(test_sentences, batchsize = B))
    GC.gc(); @time p3 = maploss(loss_v3, model, test_batches); @test p3 ≈ p2
end

@info "Timing SGD for loss_v2 and loss_v3 at various batch sizes"
train(loss, model, data) = sgd!(loss, ((model,sent) for sent in data))
@info loss_v2; test_collect = collect(test_sentences)
GC.gc(); @time train(loss_v2, model, test_collect)
for B in (1, 8, 16, 32, 64, 128, 256)
    @info loss_v3,B; test_batches = collect(LMData(test_sentences, batchsize = B))
    GC.gc(); @time train(loss_v3, model, test_batches)
end

model = NNLM(train_vocab, HIST, EMBED, HIDDEN, DROPOUT)
train_batches = collect(LMData(train_sentences))
valid_batches = collect(LMData(valid_sentences))
test_batches = collect(LMData(test_sentences))
train_batches50 = train_batches[1:50] # Small sample for quick loss calculation

epoch = adam(loss_v3, ((model, batch) for batch in train_batches))
bestmodel, bestloss = deepcopy(model), maploss(loss_v3, model, valid_batches)

progress!(ncycle(epoch, 100), seconds=5) do x
    global bestmodel, bestloss
    # Report gradient norm for the first batch
    f = @diff loss_v3(model, train_batches[1])
    gnorm = sqrt(sum(norm(grad(f,x))^2 for x in params(model)))
    # Report training and validation loss
    trnloss = maploss(loss_v3, model, train_batches50)
    devloss = maploss(loss_v3, model, valid_batches)
    # Save model that does best on validation data
    if devloss < bestloss
        bestmodel, bestloss = deepcopy(model), devloss
    end
    (trn=exp(trnloss), dev=exp(devloss), ∇=gnorm)
end

generate(bestmodel)

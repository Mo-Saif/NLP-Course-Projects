mutable struct BowNaiveBayes
    #positive train data frequency and probabilities
    pos_word2frq::Dict{String,Int64}
    pos_word2prob::Dict{String,Float64}
    pos_nwords::Int64
    pos_prior::Float64
    #negative train data frequency and probabilities
    neg_word2frq::Dict{String,Int64}
    neg_word2prob::Dict{String,Float64}
    neg_nwords::Int64
    neg_prior::Float64
    #global words2frequency dictionary on which UNK normalization will be applied
    global_word2frq::Dict{String,Int64}
    #the threshold that under of which, word frequencies are deemed UNK
    threshold::Int64
end
BowNaiveBayes(threshold::Int64 = 20) = BowNaiveBayes(Dict("UNK"=>0),Dict(),0,0.0,Dict("UNK"=>0),Dict(),0,0.0,Dict("UNK"=>0),threshold)
(N::BowNaiveBayes)(keywithlabel::Tuple{String,Int64}) = keywithlabel[2] == 1 ? get(N.pos_word2prob,keywithlabel[1],N.pos_word2prob["UNK"]) : get(N.neg_word2prob,keywithlabel[1],N.neg_word2prob["UNK"])



accuracy(y_gold,y_pred) = count(y_gold .== y_pred)/length(y_gold)

#a function-like object that tokenizes, trains, and returns predictions
function (NB::BowNaiveBayes)(train,test)
    #calculate prior probability for each review class, reviews are of format (reviewText,label)
    println("calculating prior probabilities for pos/neg training data")
    NB.pos_prior = length(trn_data[map(x -> x[2] == 1,trn_data)])/length(trn_data) 
    NB.neg_prior = length(trn_data[map(x -> x[2] == 0,trn_data)])/length(trn_data)
    
    #populating word2frequency dictionaries
    println("Tokenizing reviews...")
    countVectorizer!(NB,train)
    
    #removing words with frequncies lower than NB.threshold
    println("applying threshold to word frequencies to eliminate less informative words...")
    normalize!(NB)
    
    #calculate probabilities for each class from frequncies of words
    println("calculating words probabilities")
    train!(NB)
    
    #calculate predictions using posteriors and class proiors 
    return predict(NB,test)
    
end

function countVectorizer!(NB::BowNaiveBayes,train)
    #start populating word2frequency dictionary to use it later to populate word2probability
    for review in train
       for word in tokenizeAndClean(review[1])
            if review[2] == 1
                if haskey(NB.pos_word2frq,word)
                    NB.pos_word2frq[word] += 1
                    NB.global_word2frq[word] += 1
                else
                    NB.pos_word2frq[word] = 1
                    NB.global_word2frq[word] = 1
                end
                NB.pos_nwords += 1
            else
                if haskey(NB.neg_word2frq,word)
                    NB.neg_word2frq[word] += 1
                    NB.global_word2frq[word] += 1
                else
                    NB.neg_word2frq[word] = 1
                    NB.global_word2frq[word] = 1
                end
                NB.neg_nwords += 1
            end 
        end
    end 
end

function normalize!(NB::BowNaiveBayes)
    global_keys = keys(NB.global_word2frq)
    
    for key in global_keys
        #we don't want to remove the UNK key as in the beginning it's going to be empty
        if key == "UNK"
            continue
        end
        if NB.global_word2frq[key] < NB.threshold
            NB.global_word2frq["UNK"] += NB.global_word2frq[key]
            delete!(NB.global_word2frq,key)
        end
    end
    for key in keys(NB.pos_word2frq)
        #we don't want to remove the UNK key as in the beginning it's going to be empty
        if key == "UNK"
            continue
        end
        if !(key in global_keys)
            NB.pos_word2frq["UNK"] += NB.pos_word2frq[key]
            delete!(NB.pos_word2frq,key)
        end
    end
    for key in keys(NB.neg_word2frq)
        #we don't want to remove the UNK key as in the beginning it's going to be empty
        if key == "UNK"
            continue
        end
        if !(key in global_keys)
            NB.neg_word2frq["UNK"] += NB.neg_word2frq[key]
            delete!(NB.neg_word2frq,key)
        end
    end
end

function train!(NB::BowNaiveBayes)
    for key in keys(NB.pos_word2frq)
        NB.pos_word2prob[key]= NB.pos_word2frq[key] / NB.pos_nwords
    end
    for key in keys(NB.neg_word2frq)
        NB.neg_word2prob[key]= NB.neg_word2frq[key] / NB.neg_nwords
    end
end

#takes a string review, and returns a lowercased,tokenized and cleaned iterator of words
function tokenizeAndClean(review)
    #clean html,XML tags is if exists
    review = replace(review,r"<.*?>"=>"")
    #this rule captures phrases like state-of-the-art, or abbrivated prnouns with verbs like "they'll", "isn't" etc...
    (lowercase(i.match) for i in collect(eachmatch(r"([a-zA-Z0-9]*[-'@/]?[a-zA-Z0-9]+)+",review)))
    
end

function predict(NB,test)
    # p(Sentence|Ci) := p(word1|Ci) * p(word2|Ci) ... * p(Ci) ==OR== Sum of Log Probs
    ŷ::Array{Int64} = []
    for review in test
        
        pos_probs = [NB.([(word,1) for word in tokenizeAndClean(review[1])]); NB.pos_prior]
        neg_probs = [NB.([(word,0) for word in tokenizeAndClean(review[1])]); NB.pos_prior]
        
        pos_score = sum(log.([prob for prob in pos_probs]))
        neg_score = sum(log.([prob for prob in neg_probs]))
        
        push!(ŷ , pos_score > neg_score ? 1 : 0)
    end
    return ŷ
end

include("NaiveBayesModel.jl")

#reads list of files in a directory and returns Tuple("Review","Label -> +/-")
function readdata(pathToFiles,label)
    println("reading data...")
    data = []
#     i = 1
    for filename in readdir(pathToFiles)
        for line in eachline(pathToFiles * filename)
            push!(data,(line,label))
        end
#         i += 1
#         if i >= 2
#             break
#         end
    end
    return data
end

trn_data = [readdata("./aclImdb/train/pos/",1);readdata("./aclImdb/train/neg/",0)];
tst_data = [readdata("./aclImdb/test/pos/",1);readdata("./aclImdb/test/neg/",0)];

nbbow = BowNaiveBayes();

y_pred = nbbow(trn_data,tst_data);

y = [i[2] for i in tst_data];

println("Accuracy = $(accuracy(y,y_pred)*100)%")



# NaiveBayesBOW
Julia implementation of Bag of words using Naive Bayes method

# Notes
 * Before running the code, you need to download `AclIMDB` dataset and have it at the same directory as `NaiveBayesBOW.jl`
 * To run the script run `julia NaiveBayesBOW.jl`
 * To eliminate zero-probabilities, i used a threshold to remove words that appear less then a threshold (less informative words), and added those deleted weights to the `UNK` key.
 * The model was run with different thresholds, and as it's shown in the graph below the highest accuracy was reached with a threshold of 20
 * The file `NaiveBayesModel.jl` contains only the self-contained model code, and by just creating and object from that model (`naivebayesObj=NaiveBayes(THRESHOLD)`) and passing the threshold, all other steps would be done automatically when you execute `naivebayesObj(training_data,testing_data)`


<img width="541" alt="Screen Shot 2019-09-29 at 15 32 39" src="https://user-images.githubusercontent.com/16275685/65879487-8fe20800-e398-11e9-8f00-dd0e71411e16.png">

* The ipynb file is an alternative implementation

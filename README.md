# Novelty
DiffPool and GNNs at large have not been explored as potential methods for text summarization. However, when humans are asked to summarize a chunk of text with one phrase, the implicit exercise we do in our minds is what word most likely represents the commonality among all of the words. One way to model this is what node in a connected graph captures and can represent the subgraph of nearby nodes. DiffPool is built for this purpose and is worthwhile to pursue for the task.
In addition, as you'll see in the following section, we picked a graph-level OGB code summarization task where DiffPool has yet to be applied by the participants in the leaderboard. Furthermore, while exploring ways to outperform models on the leaderboard, we went above and beyond DiffPool implementation and tried several novel techniques to provide additional enhancements:
- CodeBERT embedding as the input into GNN in the pre-processing step
- RNN LSTM layers as post-processing
- Depth-biased pooling instead of global pooling

# Our Results vs OGBG-code2 Leaderboard
In total, 32 experiments were conducted, each taking 6 hours to code and train on a V100 GPU. We were initially puzzled at why given the same models, we could not replicate the results seen on the OGBG-code2 leaderboard. For example, we only got to at most 12.8% with GCNs, and yet the leaderboard claims 15%. The difference is more pronounced when looking at GAT (our 12.8% vs their 15.7%). What's stranger is that when we set our hyperparameters to exactly to the code linked to the GCN paper, our results were 4% less on the GCN model (our 11.1% vs their 5.0%). Same for GCN, GIn and GAT. After much exploration and review of our code, we realize that the difference lies in the hypermeter tuning. The leaderboard models were able to our more computing resources finding the optimal. Given that we have shown hyperparameters made a huge difference in our ablation studies, we guess that on average there is 3% of performance left on the table if we had all of the training resources in hand. However, given each experiment required a V100 GPU 6 hours to run, what we did was quite expensive already for a small team. It is a shame though, because it would have meant that the methodology our team tried could have achieved as high as 18%, well within the top 5 of the leaderboard.

For more details see [writeup and detailed results](https://medium.com/@nathanaelren/solving-ogb-function-name-prediction-using-data-augmentation-and-diffpool-hierarchical-pooling-fb20d3a688e1)

# Replicating results
The notebook is self contained and should run if the folder path is set correctly. Please adjust paths in Args:

      'best_model_path': "/content/gdrive/MyDrive/CS MS/Colab_Notebooks/CS224W_Final_Project/OGB_submission/best_model_ast_gs_params.pt"
      'model_path': "/content/gdrive/MyDrive/CS MS/Colab_Notebooks/CS224W_Final_Project/OGB_submission/model_params.pt"
      'eval_results_path': '/content/gdrive/MyDrive/CS MS/Colab_Notebooks/CS224W_Final_Project/OGB_submission/eval_ast_gs_results.pt'

Before running the notebook

### Using CodeBERT
Set args['is_codeBERT_encoder'] to True. Training does extend from 7 hours to 20.

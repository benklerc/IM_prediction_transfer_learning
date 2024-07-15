# IM_prediction_transfer_learning
-----------------
Pipeline to run Transfer Learning on an AlphaPeptDeep Model
-----------------
call class Transfer_Learning()
TF = Transfer_Learning()

call pipeline function pipeline_transfer_learning(training_df, prediction_df:optional)
prediction = TF.pipeline_transfer_learning(training_df, prediction_df)

Pipeline returns the prediction Dataframe(if given, else the training Dataframe) with 2 additional columns 'ccs_pred' and 'mobility_pred'
Pipeline can also save files with statistics to the provided Dataframes, as well as Evaluations of the Model at different stages in the pipeline.


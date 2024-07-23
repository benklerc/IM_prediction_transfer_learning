import os
import yaml
import pandas as pd
import numpy as np
from alphabase.psm_reader import psm_reader_provider
from peptdeep.pretrained_models import ModelManager

import matplotlib.pyplot as plt
import seaborn as sns

def load_yaml(filename) -> dict:
    with open(filename, 'r') as f:
        settings = yaml.safe_load(f)
    return settings

def save_yaml(filename, settings):
    with open(filename, "w") as file:
        yaml.dump(settings, file, sort_keys=False)


class Transfer_Learning():
    def __init__(
            self):
        '''
        Attributes
        ------------
        _settings: dict
            Dict loaded from .yaml file
        _size_train_set: int | float
            The size of the Training Set. Can either be an int as absolute value, or an float as ratio of the data set.
        _size_eval_set: int | float
            The size of the Evaluation Set. Can either be an int as absolute value, or an float as ratio of the data set.
        __random_state: int
            Random state used for the sampling process of the Evaluation and Training Set.
        _output_dir: str
            Name of the output directory in which statistical data and evaluation will be saved.
        _drop_empty_modified_sequence: bool
            If this variable is True, the process of turning the MaxQuant DF into AlphaBase Format is sped up.
        _evaluation_property: str
            Either 'ccs' or 'mobility', the column for which the evaluation should be conducted.
        model: ModelManager (AlphaPeptDeep)
            is loaded in load_model()
        training_max_quant: pd.DataFrame
            Input ms2_dataframe to pipline_transfer_learning, is manipulated in maxquant_to_alphabase.
            This Dataframe is used for Transfer Learning of the model.
        training_alpha_base: pd.DataFrame
            Dataframe ms2_max_quant in AlphaBase Format. Might contian fewer rows(maxquant_to_alphabase).
        prediction_max_quant: pd.DataFrame (optional)
            Dataframe ms1_dataframe to pipline_transfer_learning, is manipulated in maxquant_to_alphabase.
            This Dataframe will be predicted with the retrained Model.
        prediction_alpha_base: pd.DataFrame (optional)
            Dataframe ms1_max_quant in AlphaBase Format. Might contian fewer rows(maxquant_to_alphabase).
        train_set: pd.DataFrame
            Dataframe used to train the pretrained Model(self._model) on. 
            This Dataframe is a subset of ms2_alpha_base
        eval_set: pd.DataFrame
            Dataframe used to evaluate the performance of self._model. 
            This Dataframe is a subset of ms2_alpha_base, which does not overlap with train_set.
        _num_evaluation: int
            counter for the calls of evaluation()
        _num_stats: int
            counter for the calls of data_statistics()
        _num_output: int
            counter fot the calls of pipline_transfer_learning()
    
        '''
        # TODO: clean up
                # think about variable names
                # write understandable explainations to every method
                # comment nicely for better understanding

        
        # load setting from yaml
        try:
            self._settings = load_yaml('settings_tl.yaml')
        except:
            self._settings = {}
        # set default values for attributes from the yaml file
        self._size_train_set = 0.2
        self._size_eval_set = 0.2
        self.__random_state = 42
        self._output_dir = 'output_transfer_learning'
        self._drop_empty_modified_sequence = True
        self._evaluation_property = 'mobility'
        # check if yaml file overwrites default settings
        if 'size_train_set' in self._settings:
            self._size_train_set = self._settings['size_train_set']
        if 'size_eval_set' in self._settings:
            self._size_eval_set = self._settings['size_eval_set']
        if 'random_state' in self._settings:
            self.__random_state = self._settings['random_state']
        if 'output_directory' in self._settings:
            self._output_dir = self._settings['output_directory']
        if 'drop_empty_modified_sequence' in self._settings:
            self._drop_empty_modified_sequence = self._settings['drop_empty_modified_sequence']
        if 'evaluation_property' in self._settings:
            self._evaluation_property = self._settings['evaluation_property']

        # introduce attributes
        self.model = None
        self.training_max_quant = pd.DataFrame() 
        self.training_alpha_base = pd.DataFrame() 
        self.prediction_max_quant = None
        self.ms1_alpha_base = None    
        self.train_set = pd.DataFrame() # AlphaBase Format
        self.eval_set = pd.DataFrame() # AlphaBase Format        
        self._num_evaluation = 0 # counter variable to keep track of evaluations
        self._num_stats = 0 # counter variable to keep track of statistics
        self._num_pipeline = 0 # counter variable to keep track of how often the pipeline was run
        

    def pipeline_transfer_learning(self, 
                trainings_dataframe: pd.DataFrame | tuple, 
                prediction_dataframe: pd.DataFrame| tuple =None, 
                statistics: bool= True, 
                additional_statistics: str | list = None, 
                additional_mappings: dict = None,
                custom_merging_list: list = None )-> pd.DataFrame:
        '''
        Wrapper function to run whole pipeline for transfer learning
        Returns Dataframe(if given prediction_df, else trainings_df) with 'ccs_pred and 'mobility_pred' as columns
        -------------------
        Parameters
        -------------------
        trainings_dataframe: pd.DataFrame | tuple(filepath, seperation)
                dataframe which contains IM data to train
                tuple example: (dataframe.txt, '\t')
        prediction_dataframe: pd.DataFrame | tuple (filepath, seperation)
                does not need to contain IM Data, Prediction will be made on this dataframe if provided
        statistics: bool
                if statistical information about both dataframes should be provided in output_dir
        additional_statistics: str | list
                statistics which should be provided additional to ['mods', 'charge', 'nAA', 'precursor_mz', 'ccs', 'mobility']
                in AlphaBase format
        additional_mappings: dict
            Additional mappings from AlphaBase column names to MaxQuant columnnames.
            Is used in merge_alphabase_to_maxquant
            Can also overwrite mappings in the mapping_dict.
        costum_merging_list: list
            This list overwrites the columns on which the merge from prediction to the input Dataframe will be done.
        '''
        # create right output directory(in case of repeated use of pipline function)
        if self._num_pipeline > 0:
            self._output_dir = f"{self._settings['output_directory']}_{self._num_pipeline}"
        self._num_pipeline +=1
        # reset counters for evaluation and statistics(in case of repeated use of pipline function)
        self._num_evaluation = 0
        self._num_stats = 0

        # check the category is as expected
        if not (self._evaluation_property == 'ccs') | (self._evaluation_property == 'mobility'):
            raise Exception('The category for the evaluation is not available')

        # load model
        self.load_model()

        # save ms2 dataframes
        if type(trainings_dataframe) is pd.DataFrame:
            self.training_max_quant = trainings_dataframe
        elif type(trainings_dataframe) is tuple:
            self.training_max_quant = pd.read_csv(trainings_dataframe[0], sep = trainings_dataframe[1] )
        self.training_alpha_base = self.maxquant_to_alphabase(self.training_max_quant)

        # save ms1 dataframes (if given)
        if prediction_dataframe is not None:
            if type(prediction_dataframe) is pd.DataFrame:
                self.prediction_max_quant = prediction_dataframe
            elif type(prediction_dataframe) is tuple:
                self.prediction_max_quant = pd.read_csv(prediction_dataframe[0], sep = prediction_dataframe[1] )
            self.ms1_alpha_base = self.maxquant_to_alphabase(self.prediction_max_quant)

        # Calculate Statistics, if wanted
        if statistics:
            self.data_statistics(self.training_alpha_base,additional_statistics)
            if self.prediction_max_quant is not None:
                self.data_statistics(self.ms1_alpha_base, additional_statistics)

        #Samples Data into Training and Evaluation Set
        self.sample_data(self.training_alpha_base)

        # Run Model on Evaluation Data
        prediction_eval_1 = self.run_model(self.eval_set)

        # Evaluate untrained Model on Evaluation Data
        self.data_evaluation(prediction_eval_1)

        # Retrain Model
        self.transfer_learning_model()

        # Run retrained Model on Evaluation Data
        prediction_eval_2 = self.run_model(self.eval_set)

        # Evaluate retrained Model on Evaluation Data
        self.data_evaluation(prediction_eval_2)

        # Run Model on dataframe(train or prediction)
        # merge prediction into MaxQuant DF
        if prediction_dataframe is not None:
            prediction_data = self.run_model(self.ms1_alpha_base) 
            if custom_merging_list is not None:
                custom_merging_list.append(['Sequence', 'Charge', 'Score', 'Length', 'Retention time', 'Proteins', 'Gene names', 'MS/MS scan number', 'Raw file'
                                                  , 'm/z', 'Intensity' ])
            else:
                custom_merging_list = ['Sequence', 'Charge', 'Score', 'Length', 'Retention time', 'Proteins', 'Gene names', 'MS/MS scan number', 'Raw file'
                                                  , 'm/z', 'Intensity' ]
            prediction_max_quant = self.merge_alphabase_to_maxquant(prediction_data, self.prediction_max_quant, additional_mappings, custom_merging_list)
        else:
            prediction_data= self.run_model(self.training_alpha_base) 
            # Evaluate model on whole dataset
            self.data_evaluation(prediction_data)
            prediction_max_quant = self.merge_alphabase_to_maxquant(prediction_data, self.training_max_quant, additional_mappings, custom_merging_list)
        
        # write used settings in output_directory
        save_yaml(f'{self._output_dir}/settings.yaml', self._settings)

        # return prediction of the retrained model
        return prediction_max_quant
    
    def maxquant_to_alphabase(self, dataframe: pd.DataFrame):
        '''
        Changes the DataFrame from MaxQuant format to AlphaBase format.
        Introduces 'mods', 'mod_site', 'Original index' columns
        By Default all rows with empty modified sequence will be dropped. It can be choosen to try to recreate as many modified sequences as possible by 
            setting self._drop_empty_modified_sequences to False
        ------------
        Parameters
        ------------
        dataframe: pd.DataFrame
            Dataframe in MaxQuant format
        '''
        if not self._drop_empty_modified_sequence:
            # recreate as many modified sequences as possible
            dataframe  = self.fill_modified_sequence(dataframe)
                    
        # drop all sequences which alphabase cannot convert
        dataframe = dataframe.dropna(subset = ['Modified sequence'])
        dataframe.loc[:,'MS/MS scan number'] = dataframe['MS/MS scan number'].fillna(-1)
        # introduce original index column to be able to merge alpha_base_df back into their original max_quant_df
        dataframe.loc[:,'Original index'] = dataframe.index
        # run AlphaBase
        mq_reader = psm_reader_provider.get_reader('maxquant')
        mq_reader.column_mapping['Original index'] = 'Original index'
        mq_reader._translate_columns(dataframe)
        mq_reader._transform_table(dataframe)
        mq_reader._translate_decoy(dataframe)
        mq_reader._translate_score(dataframe)
        mq_reader._load_modifications(dataframe)
        mq_reader._translate_modifications()
        mq_reader._post_process(dataframe)  
        return mq_reader.psm_df
    
    def load_model(self):
        '''
        Load the model on which Transfer Learning is performed
        '''
        self.model = ModelManager()
        self.model.load_installed_models()

    def sample_data(self, dataframe: pd.DataFrame):
        '''
        Splits Data for Training and Evaluation Set.
        ----------------
        Parameters
        ----------------
        dataframe: pd.DataFrame
            Dataframe from which Training set and Evaluation set is taken
        '''
        # get absolute values for the set sizes
        if type(self._size_eval_set) is float:
            self._size_eval_set = round(self._size_eval_set*len(self.training_alpha_base))
        if type(self._size_train_set) is float:
            self._size_train_set = round(self._size_train_set*len(self.training_alpha_base))

         # if only one df provided, train und eval set are not allowed to be the full dataset
        if (self.prediction_max_quant is None) &( (self._size_eval_set + self._size_train_set) >= len(self.training_max_quant)):
            raise Exception('Your training and evaluation sets are too big')
        
        # sample training set
        self.train_set = dataframe.sample(n=self._size_train_set, random_state=self.__random_state)

        # make sure no overlap between the sets
        df_not_train = dataframe.drop(self.train_set.index)
        # sample evaluation set
        self.eval_set = df_not_train.sample(n=self._size_eval_set, random_state=self.__random_state)


    def data_statistics(self, dataframe: pd.DataFrame, addiotional_statistics: (str | list )= None):
        '''
        Saves basic statistics in folder ../data_statistics
        ---------------
        dataframe: pd.DataFrame
            should be in AlphaBase format
        additional_statistics: str|list
            Statistics which should also be saved, additional to statistics
        '''
        statistics = ['mods', 'charge', 'nAA', 'precursor_mz', 'ccs', 'mobility']
        if addiotional_statistics:
            statistics.append(addiotional_statistics)

        # get the naming of the files according to the function calls in pipline_transfer_learning
        if self._num_stats ==0:
            name = 'training_dataframe'
        elif self._num_stats == 1:
            name = 'prediction_dataframe'
        else:
            name = f'{self._num_stats}'
        #check if directory already exists
        if not os.path.exists(f'{self._output_dir}/data_statistics_{name}'):
            os.makedirs(f'{self._output_dir}/data_statistics_{name}')

        # plot histograms for every statistic and save them as .png
        for stat in statistics:
            if stat in dataframe.columns:
                self.plot_histograms(df = dataframe,column = stat, directory = f'{self._output_dir}/data_statistics_{name}')

        self._num_stats +=1
        


    def run_model(self, dataframe: pd.DataFrame):
        '''
        Runs the Prediction of CCS/IM with AlphaPeptDeep
        -----------
        dataframe: pd.DataFrame
            should be in AlphaBase format
        '''
        prediction = self.model.predict_mobility(dataframe)
        return prediction

    def data_evaluation(self, prediction: pd.DataFrame):
        '''
        prediction: pd.DataFrame in AlphaBaseFormat
        '''
        # check if directory exists
        if not os.path.exists(f'{self._output_dir}/evaluation'):
            os.makedirs(f'{self._output_dir}/evaluation')
        
        # introduce new column with prediction error
        prediction[f'{self._evaluation_property}_error'] = prediction[self._evaluation_property] - prediction[f'{self._evaluation_property}_pred']
        # calculate evaluation values
        error_mean = prediction[f'{self._evaluation_property}_error'].mean()
        error_std = prediction[f'{self._evaluation_property}_error'] .std()
        perc_low = np.percentile(prediction[f'{self._evaluation_property}_error'], 2.5)
        perc_up = np.percentile(prediction[f'{self._evaluation_property}_error'], 97.5)
        perc_half = (abs(perc_low) + abs(perc_up))/2
        
        # name the evaluation files accordingly to the function calls from pipline_transfer_learning
        # write the evalaution values to .txt file
        if self._num_evaluation == 0:
            f = open(f"{self._output_dir}/evaluation/evaluation_{self._evaluation_property}_before_TF", "w")
            f.write(f'Evaluation before Transfer Learning on Evaluation Set\n')
        elif self._num_evaluation == 1:
            f = open(f"{self._output_dir}/evaluation/evaluation_{self._evaluation_property}_after_TF", "w")
            f.write(f'Evaluation after Transfer Learning on Evaluation Set\n')
        elif (self._num_evaluation == 2) & (self.prediction_max_quant is None):
            f = open(f"{self._output_dir}/evaluation/evaluation_{self._evaluation_property}_whole_data", "w")
            f.write(f'Evaluation after Transfer Learning on whole Data Set\n')
        else: 
            f = open(f"{self._output_dir}/evaluation/evaluation_{self._evaluation_property}_additional_{self._num_evaluation}", "w")
            f.write(f'Evaluation {self._num_evaluation}: Additional Evaluation\n')
        f.write(f"Error mean:{error_mean}\nError standard deviation:{error_std}\nDelta 95:({perc_low}, {perc_up})\n\tHalf Range:{perc_half}")
        f.close()
        
        self._num_evaluation += 1

    def transfer_learning_model(self):
        '''
        Transfer Learning of model
        '''
        self.model.train_ccs_model(self.train_set) # TODO: inplace??

    def fill_modified_sequence(self, df:pd.DataFrame):
        '''
            Manipulates df['Modified sequence]
            if it is NaN
            Unmodified -> _Sequence_
            Acetly -> _Acetyl(N term)Sequence_
            Oxidation -> _Sequence.replace('M', 'M(Oxidation (M)))_ (Only if #Oxidations == Appearances of Methionine)
        '''
        for index, row in df.iterrows():
            if pd.isnull(row['Modified sequence']):
                if row['Modifications'] == 'Unmodified':
                    df.loc[index,'Modified sequence'] = f'_{row["Sequence"]}_'
                else:
                    if row['Acetyl (Protein N-term)'] == 1 :
                        df.loc[index,'Modified sequence'] = f'_(Acetyl (Protein N-term)){row["Sequence"]}_'
                    if row['Oxidation (M)'] > 0:
                        if row['Oxidation (M)'] == row['Sequence'].count('M'): # Check if all Mehtionine are Oxidated#
                            if pd.isnull(df.loc[index,'Modified sequence']): # Check if a string was added before
                                df.loc[index,'Modified sequence'] = f"_{row['Sequence'].replace('M', 'M(Oxidation (M))')}_"
                            else:
                                df.loc[index,'Modified sequence'] = f"{df.loc[index,'Modified sequence'].replace('M', 'M(Oxidation (M))')}"
        return df

    def plot_histograms(self, df: pd.DataFrame, column: str, directory: str):
        '''
        plots histograms for the statistics and saves them as .png
        '''
        description = column
        description = column.replace('/', '_')# might not work, might need if statement if str.contains('\')              

        if df[column].dtype == object:         
            #plt.figure(figsize=(8,12))   
            p = sns.histplot(df[column], discrete=True).get_figure()
            plt.xticks(rotation = 90)
        elif  df[column].dtype ==  'int64':
            if len(df[column].unique())>20:
                plt.figure(figsize=(len(df[column].unique())/2,8)) 
            p = sns.histplot(pd.Categorical(df[column])).get_figure()
            plt.xticks(ticks=df[column].unique())
        else:
            p = sns.histplot(df[column], discrete=False).get_figure()

        plt.title(description)
        plt.xlabel(column)
        p.savefig(f'{directory}/{description}',bbox_inches="tight")            
        plt.close()


    def merge_alphabase_to_maxquant(self, alpha_base_df:pd.DataFrame, max_quant_df:pd.DataFrame, additional_mappings: dict = None, costum_merging_list: list= None)-> pd.DataFrame:
        '''
        merges AlphaBase format DF into a df in MaxQuant format

        Parameters:
        --------------
        alpha_base_df: pd.DataFrame
            DF in AlphaBase format.
        max_quant_df: pd.DataFrame
            DF in MaxQuant format.
        additional_mappings: dict
            Mappings which are not contained in the mapping dict to rename more columns in AlphaBase Dataframe.
            Can also overwrite mappings in the mapping_dict.
        costum_merging_list: list
            This list overwrites the columns on which the merge will be done.
        '''
        mapping_dict = {
            'sequence': 'Sequence',
            'charge': 'Charge',
            'rt': 'Retention time',
            'ccs': 'CCS',
            'mobility': '1/K0',
            'scan_num': 'MS/MS scan number',
            'raw_name': 'Raw file',
            'precursor_mz': 'm/z',
            'score': 'Score',
            'proteins': 'Proteins',
            'genes': 'Gene names',
            'decoy': 'Reverse',
            'intensity': 'Intensity',
            'nAA':'Length'}
        
        merging_list = ['Sequence', 'Charge', 'CCS', 'Score', 'Length', 'Retention time', 'Proteins', 'Gene names','1/K0' , 'MS/MS scan number', 'Raw file'
                                                  , 'm/z', 'Intensity' ]
        if costum_merging_list is not None:
            merging_list = costum_merging_list
        # update mapping dict, with possible user additions
        if additional_mappings is not None:
            mapping_dict.update(additional_mappings)
        # rename the columns of the alphabase dataframe according to the mapping dict
        alpha_base_df.rename(columns=mapping_dict, inplace=True)
        # set the original index as index
        alpha_base_df.set_index('Original index', inplace=True)
        df_merged = pd.merge(max_quant_df, alpha_base_df, on = merging_list, how = 'inner')
        return df_merged


    




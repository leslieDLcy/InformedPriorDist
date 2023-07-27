import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from pipeline import ensemble_predict
import pandas as pd
import tqdm
from modules import CI95_interval
from modules import acc_metrics

Forecast_val = namedtuple("Forecast_val", ["ground_truth", "a_forecast", "mae_value"])
Forecast_Val_Ensemble = namedtuple("Forecast_Val_Ensemble", ["ground_truth", "ensemble_val_fcst", "mean_mae"])


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                             train_df, val_df,
                             label_columns=None):
    
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df

        """ `label_colunmns` is used to select interested features to use
        If None, then use all, else use selected, in the form of a 'list';
        """
        # Work out the label column indices.
        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                                     enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])


    def __str__(self):
        return '\n'.join([
                'Hint: when you create a window object, then the mapping pair is determined, \
                meaning that the way the model trains and predicts is determined. ie, (*, time steps, features) -> label',
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])
        


    def split_window(self, features):
        """a step that, given a window, splits training part and the label part, 
        i.e. training window -> label window;
        This func works with multiple features;

        Parameters:
        -----------
        features : xxx
        """

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns], 
                    axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels


    def make_dataset(self, data):
        """
        The real deal. It runs the func `self.split_window` within.
        Given a time series dataframe, convert it to a `tf.data.Dataset` of (input_window, label_window)

        Parameters:
        -----------
        We will use data = train_df
        """

        data = np.array(data, dtype=np.float32)
        # create `tf.data.Dataset` object
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = data,
            targets=None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle=True,
            batch_size=64,
        )
        ds = ds.map(self.split_window)
        return ds
    

    @property
    def train(self):
        return self.make_dataset(self.train_df)


    @property
    def val(self):
        return self.make_dataset(self.val_df)


    @property
    def example(self):
        """Get and cache an example batch of 'inputs, labels' for plotting.
        Let the user know the shape of window they'are using    
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example was found, so get one from the '.train' dataset
            result = next(iter(self.train))
            # and cache it
            self._example = result
        return result
    

    ##########################################
    ########## validation ####################
    ##########################################


    def cp_val_B(self, model, ensemble_size, style='mc_dropout'):
        """ for svi models """
        
        self.all_normalized_data =  pd.concat([self.train_df, self.val_df])
        all_normalized_data = self.all_normalized_data.values

        # deterministic lstm
        # preds = np.stack([model(tf.expand_dims(all_normalized_data[i-self.window.input_width:i, :], axis=0)) for i in range(200, 208)])
        # preds = np.squeeze(preds)

        # bayesian lstm
        ensemble_results = []

        if style == 'svi':
            for num in tqdm.tqdm(range(ensemble_size)):
                pred = np.stack([model(tf.expand_dims(all_normalized_data[i-self.input_width:i, :], axis=0)).mean().numpy() for i in range(200, 208)])
                ensemble_results.append(np.squeeze(pred))

        elif style == 'mc_dropout':
            for num in tqdm.tqdm(range(ensemble_size)):
                pred = np.stack([model(tf.expand_dims(all_normalized_data[i-self.input_width:i, :], axis=0), training=True) for i in range(200, 208)])
                ensemble_results.append(np.squeeze(pred))

        ensemble_results = np.vstack(ensemble_results)

        # compute the  MAE
        MAEs = tf.keras.metrics.mean_absolute_error(
            y_true = self.val_df['revenue'], y_pred=ensemble_results)
        
        MAPEs = tf.keras.metrics.mean_absolute_percentage_error(
            y_true = self.val_df['revenue'], y_pred=ensemble_results)
        
        mae, mape = np.mean(MAEs), np.mean(MAPEs)

        return ensemble_results, acc_metrics(mae=mae, mape=mape)
        
        
    def cp_val_lstm_mcdropout_oneshot(self, model):
        all_normalized_data =  pd.concat([self.train_df, self.val_df])
        all_normalized_data = all_normalized_data.values

        preds = np.stack([model(tf.expand_dims(all_normalized_data[i-self.input_width:i, :], axis=0), training=True) for i in range(200, 208)])
        preds = np.squeeze(preds)

        # compute the  MAE
        mae =  tf.keras.metrics.mean_absolute_error(
            y_true = self.val_df['revenue'], y_pred=preds)
        
        mape = tf.keras.metrics.mean_absolute_percentage_error(
            y_true=self.val_df['revenue'], 
            y_pred=preds)
        return acc_metrics(mae=mae, mape=mape)



    def display_lstm_val(self, ensemble_results, val_x_axis):
        """ display epistemic uncertainty for lstm models """

        fig, ax = plt.subplots(figsize=(12, 4))

        # plot the train and val [revenue] series 
        ax.plot(self.all_normalized_data.revenue, marker='+')

        # boundry between train and val
        ax.axvline(x=200, ymin=0, ymax=1, color='purple', linestyle='--')

        CI95_metric = CI95_interval(ensemble_results)

        # add the predictions then 
        ax.scatter(val_x_axis, self.val_df['revenue'], color='r', marker='o', )
        # ax.plot(val_x_axis, ensemble_results.mean(axis=0), color='r', label='mean', linewidth=0.5)
        # ax.plot(val_x_axis, CI95_metric.PI_median, color='r', label='median', linewidth=0.5)

        ax.fill_between(val_x_axis,
                CI95_metric.PI_2p5,
                CI95_metric.PI_97p5,
                color='salmon', 
                alpha=0.2, 
                label='95CI')

        ax.legend(loc='best')
        ax.grid(linestyle=':')
        ax.set_xlabel('time index')
        ax.set_ylabel('Revenue in million (standardized))')
        ax.set_title('Epistemic Uncertainty for LSTM model')
        ax.set_xlim([160,208])



    def illustrate_val(self, model, ensemble_size):
        """ plot the predictions during val range """

        combined_df = pd.concat([self.train_df, self.val_df])

                # create `tf.data.Dataset` object
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = combined_df,
            targets=None,
            sequence_length = self.input_width,
            sequence_stride = 1,
            shuffle=False,
            batch_size=1,
        )

        all_data = np.concatenate([x for x in ds], axis=0)
        val_illus_df = all_data[-8:, :, :]

        self.enPred_WholeTestSet = ensemble_predict(
            model=model, 
            test_data = val_illus_df, 
            ensemble_size=ensemble_size)

        return self.enPred_WholeTestSet



    ''' below are library codes written before '''

    def val_fcast(self, data, model):
        """ Predict along with the `validation` time series;
        
        This func provides a shortcut to the validation of deterministic model and window pair;

        call signature
        --------------
        ```
        forecastDNN_shortcut = w1.val_fcast(val_df, DNN_model)
        ```

        Hint
        ----
        if only want the results for evaluation, then `model.evaluate()` will suffice.
        Instead, this function provides the forecasts so that a time domain plot can thus be made.
        """
        data_values = np.array(data, dtype=np.float32)
        
        # create `tf.data.Dataset` object
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = data_values,
            targets=None,
            sequence_length = self.input_width,
            sequence_stride = 1,
            shuffle=False,
            batch_size=32,
        )

        # determine the shape of pred when given a 1D time series
        num_of_predictions = len(data) - self.input_width
        gt = np.squeeze(data[self.label_columns][(-num_of_predictions):])

        forecast = model.predict(ds)
        forecast = np.squeeze(forecast[:-1, ...])
     
        # print("gt shape:", gt.shape)
        # print("forecast shape:", forecast.shape)

        mae = tf.keras.metrics.mean_absolute_error(gt, forecast).numpy()
        return Forecast_val(gt, forecast, mae)















    # def val_fcast_mc_DNN(self, data, model, ensemble_size):
    #     """ This func provides a shortcut to the validation of model and window pair """

    #     data_values = np.array(data, dtype=np.float32)
        
    #     # create `tf.data.Dataset` object
    #     ds = tf.keras.utils.timeseries_dataset_from_array(
    #         data = data_values,
    #         targets=None,
    #         sequence_length = self.input_width,
    #         sequence_stride = 1,
    #         shuffle=False,
    #         batch_size=32,
    #     )

    #     # determine the shape of pred when given a 1D time series
    #     num_of_predictions = len(data_values) - self.input_width
    #     gt = np.array(np.squeeze(data[self.label_columns][-num_of_predictions:]))
        
    #     MC_forecasts = []
    #     for _ in track(range(ensemble_size)):
    #             forecast = model.predict(ds)
    #             forecast = np.squeeze(forecast[:-1, ...])
    #             MC_forecasts.append(forecast)
    #     MC_forecasts = np.vstack(MC_forecasts)

    #     mae = tf.keras.metrics.mean_absolute_error(gt, np.mean(MC_forecasts, axis=0)).numpy()
    #     return Forecast_Val_Ensemble(gt, MC_forecasts, mae)


    # def val_fcast_mc_LSTM(self, data, model, ensemble_size=100):
    #     """ Validation in the case of using LSTM Dropout model in ensemble setting;
        
    #     Hint:
    #     -----
    #     A MC implementation of 'val_forecast' function.
    #     I bind this function here because the validation should be compatible with the window.
        
    #     Parameters:
    #     -----------
    #     Becasue we will use model(x, training=False) so that we cannot batch the input;
    #     It means the window shape and label column has been set equally with the first window function.
    #     """

    #     data_values = np.array(data, dtype=np.float32)
        
    #     # create `tf.data.Dataset` object
    #     ds = tf.keras.utils.timeseries_dataset_from_array(
    #         data = data_values,
    #         targets=None,
    #         sequence_length = self.input_width,
    #         sequence_stride = 1,
    #         shuffle=False,
    #         batch_size=1,
    #     )
        
    #     MC_forecasts = []
    #     for _ in track(range(ensemble_size)):
    #             dataset_lst = list(ds.as_numpy_iterator())
    #             dataset_input_arrays = np.concatenate(dataset_lst, axis=0)
    #             one_forecast_LSTM = model(dataset_input_arrays, training=True)
    #             one_forecast_LSTM = np.squeeze(one_forecast_LSTM[:-1, ...])
    #             MC_forecasts.append(one_forecast_LSTM)
    #     MC_forecasts = np.vstack(MC_forecasts)
        
    #     # determine the shape of pred when given a 1D time series
    #     num_of_predictions = len(data_values) - self.input_width
    #     gt = np.array(np.squeeze(data[self.label_columns][-num_of_predictions:]))
        
    #     mae = tf.keras.metrics.mean_absolute_error(gt, np.mean(MC_forecasts, axis=0)).numpy()
    #     return Forecast_Val_Ensemble(gt, MC_forecasts, mae)



    # @staticmethod
    # def val_ensemble_plot(Forecast_Val_Ensemble, model_name, option):
    #     """ 
    #     Once the ensemble forecast on validation set is obtained
    #     typically saved in a variable called `MC_forecast_DNN`;
    #     we then plot the uncertainty using either
    #     "mean and std" or "95CI"

    #     Parameters:
    #     ----------
    #     Forecast_Val_Ensemble : an ensemble forecast on validation data object;
    #     model_name: which model to use (DenseVariational or Flipout or...);
    #     option : style of the interval ("mean and std" or "95CI");

    #     """
    #     # get the gt, a hint below
    #     # np.array(val_df.loc[:,w1.label_columns][-14015:].shape
    #     dummny_xaxis= np.arange(len(Forecast_Val_Ensemble.ground_truth))
    #     plt.figure(figsize=(10, 6))
    #     if option == 'meanstd':
    #         # PSD uncertainty plot #1
    #         ms_metrics = mean_std_np(Forecast_Val_Ensemble.ensemble_val_fcst)
    #         # the mean curve
    #         plt.plot(dummny_xaxis, ms_metrics['mean'], color='red', label='Predictive mean', linewidth=0.5)
    #         plt.fill_between(dummny_xaxis,
    #                 ms_metrics['mean'] + 2 * ms_metrics['sigma'], 
    #                 ms_metrics['mean'] - 2 * ms_metrics['sigma'],
    #                 color='salmon', 
    #                 alpha=0.2, 
    #                 label='Mean +- 2 sigma')

    #         # and the target PSD
    #         plt.plot(dummny_xaxis, Forecast_Val_Ensemble.ground_truth, label='the ground truth', linewidth=0.5)
    #         plt.xlabel('dummy xaxis', fontsize=12)
    #         plt.ylabel('prediction', fontsize=12)
    #         plt.title(f'Predictive uncertainty represented by mean and std by {model_name}', 
    #                 fontsize=12)
    #         plt.grid()
    #         plt.legend()
    #     elif option == 'CI95':
    #         # PSD uncertainty plot #2 -> (2.5%, median, 97.5%) plot
    #         CI95_metrics = CI95_interval(Forecast_Val_Ensemble.ensemble_val_fcst)
    #         # the median
    #         plt.plot(dummny_xaxis, CI95_metrics['PI_median'], color='red', linewidth=0.5, label='median')
    #         plt.fill_between(dummny_xaxis, 
    #                 CI95_metrics['PI_2p5'],
    #                 CI95_metrics['PI_97p5'],
    #                 color='coral',
    #                 alpha=0.2,
    #                 label='95% CI')
    #         # and the target PSD
    #         plt.plot(dummny_xaxis, Forecast_Val_Ensemble.ground_truth, label='the ground truth', linewidth=0.5)
    #         plt.xlabel('dummy xaxis', fontsize=12)
    #         plt.ylabel('prediction', fontsize=12)
    #         plt.title(f'Predictive uncertainty on validation data by {model_name}', fontsize=12)
    #         plt.grid()
    #         plt.legend()
    
    
    def hint_shape(self):
    
        """a helper function to know what shape of mapping used"""
        print(self.train.element_spec)

import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple
from pipeline import ensemble_predict
tfd = tfp.distributions

PI_bound = namedtuple(
    'PI_bound', ['ensembleAverage', 'lowerBound', 'upperBound', 'mean', 'std'])
CI95_metrics = namedtuple("CI95_metrics", ["PI_2p5", "PI_97p5", "PI_median"])

# create an universal metric to compare the performance of different models
acc_metrics = namedtuple('acc_metrics', ['mae', 'mape'])


class EnsemblePredict():

    def __init__(self, ensemble_size, test_features, test_labels):
        self.ensemble_size = ensemble_size
        self.test_features = test_features
        self.gt = test_labels

    def mc_predict_test_set(self, model):

        self.enPred_WholeTestSet = ensemble_predict(
            model=model,
            test_data=self.test_features,
            ensemble_size=self.ensemble_size)

    def pl_epistemic(self, val_x_axis):
        """ plot epistemic uncertainty of the whole test set """

        pl_preds_uncertainty(
            x_axis=val_x_axis,
            predictions=self.enPred_WholeTestSet,
            ground_truth=self.gt,
            option='CI95')

    def pl_residual_B(self, low=2, limit=4):

        mean = np.mean(self.enPred_WholeTestSet, axis=0)

        fig, ax = plt.subplots()
        ax.scatter(mean, self.gt, color='blue', alpha=0.5)

        ax.plot(np.arange(low, limit, 0.01), np.arange(
            low, limit, 0.01), color='gray', ls='--')

        error = np.std(self.enPred_WholeTestSet, axis=0)
        ax.errorbar(mean, self.gt,  xerr=error,
                    fmt='none', ecolor='blue', alpha=0.5)
        ax.set_xlabel('Predicted revenue in million')
        ax.set_ylabel('Ground truth revenue in million')

    def cp_ensemble_metrics(self,):
        """ compute the uncertainty metrics for val data set"""

        # compute the  MAE
        MAEs = tf.keras.metrics.mean_absolute_error(
            y_true=self.gt, y_pred=self.enPred_WholeTestSet)

        MAPEs = tf.keras.metrics.mean_absolute_percentage_error(
            y_true=self.gt,
            y_pred=self.enPred_WholeTestSet)

        mae = np.mean(MAEs)
        mape = np.mean(MAPEs)
        return acc_metrics(mae=mae, mape=mape)

    def pl_epistemic_ts(self, dataset):
        """ plot epistemic uncertainty of the whole test set """

        fig, ax = plt.subplots(figsize=(12, 4))

        # plot the full train and val [revenue] series
        ax.plot(dataset.revenue, marker='+')

        # boundry between train and val
        ax.axvline(x=200, ymin=0, ymax=1, color='purple', linestyle='--')

        # plot the val results
        val_x_axis = np.arange(200, 208)

        # add the ground truth
        ax.scatter(val_x_axis, self.gt, color='r', marker='o', )

        CI95_metric = CI95_interval(self.enPred_WholeTestSet)

        # the ensemble median
        # ax.plot(val_x_axis, CI95_metric.PI_median, color='red', label='ensemble median')

        ax.fill_between(val_x_axis,
                        CI95_metric.PI_2p5,
                        CI95_metric.PI_97p5,
                        color='coral',
                        alpha=0.2,
                        label=r'95\%' ' credible interval')
        ax.legend(loc='best')
        ax.grid(linestyle=':')
        ax.set_ylabel('Revenue in million')
        ax.set_title('Epistemic uncertainty')
        ax.set_xlim([160, 208])

    def mixed_uncertainty_PI(self, dataset, PI_dp_bounds_all_list, val_x_axis):
        """ plot epistemic uncertainty of the whole test set """

        fig, ax = plt.subplots(figsize=(12, 4))

        # plot the full train and val [revenue] series
        ax.plot(dataset.revenue, marker='+')

        # boundry between train and val
        ax.axvline(x=200, ymin=0, ymax=1, color='purple', linestyle='--')

        # add the ground truth
        ax.scatter(val_x_axis, self.gt, color='r', marker='o', )

        mean_curve = np.array([x.mean[0] for x in PI_dp_bounds_all_list])
        lb_curve = np.array([x.lowerBound[0] for x in PI_dp_bounds_all_list])
        ub_curve = np.array([x.upperBound[0] for x in PI_dp_bounds_all_list])

        # the ensemble median
        ax.plot(val_x_axis, mean_curve, color='red', label='ensemble mean')

        ax.fill_between(val_x_axis,
                        lb_curve,
                        ub_curve,
                        color='coral',
                        alpha=0.2,
                        label=r'95\%' ' credible interval')
        ax.legend(loc='best')
        ax.grid(linestyle=':')
        ax.set_title('Mixed uncertainty - both aleatoric and epistemic')
        ax.set_xlim([160, 208])

    def cp_ensemble_testset(self, model):
        """ compute the tfd distribution objects (with mean and stddev) for all the data points
        """
        container = []
        # compute the ensemble dist for all data points
        for i in tqdm.tqdm(range(len(self.gt))):
            dp_dist_result = cp_ensemble_dp(
                model=model,
                input_dp=self.test_features[i][np.newaxis],
                ensemble_size=self.ensemble_size)
            container.append(dp_dist_result)
        self._dist_obj_testset = container

    def get_PI_bounds_testset(self, style='gmm'):
        """ choose a PI bound style and get these bounds for the whole test set 
        ! the current entry point 

        Parameters
        ----------
        style : str,
            two choices: 'gmm' and 'envelop'
        """

        # if not hasattr(self, '_dist_obj_testset'):
        #     self.cp_ensemble_testset()

        PI_dp_bounds_all_list = [cp_dp_PI_bound(
            ensemble_dp_dist, style=style) for ensemble_dp_dist in self._dist_obj_testset]
        return PI_dp_bounds_all_list

    def cp_d_metrics(self,):
        """ compute the deterministic metrics for val data set"""

        # compute the  MAE
        MAEs = tf.keras.metrics.mean_absolute_error(
            y_true=self.gt, y_pred=self.enPred_WholeTestSet)

        mape = tf.keras.metrics.mean_absolute_percentage_error(
            y_true=self.gt,
            y_pred=self.enPred_WholeTestSet)

        mae = np.mean(MAEs)
        mape = np.mean(mape)
        return mae, mape


class AleatoricPredict:
    """ for aleatoric model """

    def __init__(self, test_features, test_labels):
        self.test_features = test_features
        self.test_labels = test_labels


    def predict_dist(self, model, data=None):
        """ predict the distribution object """

        if data is None:
            data = self.test_features
        
        # compute the mean of the conditional distribution objects
        self.conditional_means = model(data).mean()

        # compute the variance of the dist objects
        self.conditional_stds = model(data).stddev()


    @property
    def lower_bound(self):
        return self.conditional_means - 2 * self.conditional_means


    @property
    def upper_bound(self):
        return self.conditional_means + 2 * self.conditional_means
    

    def cp_metrics(self,):
        """ currently we compute mae and mape """

        mae = tf.keras.metrics.mean_absolute_error(
            y_true=self.test_labels, y_pred=np.squeeze(self.conditional_means))
        
        mape = tf.keras.metrics.mean_absolute_percentage_error(
            y_true=self.test_labels, 
            y_pred=np.squeeze(self.conditional_means))

        return acc_metrics(mae=mae, mape=mape)


    def pl_aleatoric_uncertainty(self, val_x_axis):
        """ plot the aleatoric uncertainty """

        # val range only
        fig, ax = plt.subplots()

        # ground truth
        ax.scatter(val_x_axis, self.test_labels, marker='o', alpha=0.4, label='ground truth')

        # mean prediction
        ax.plot(val_x_axis, self.conditional_means, color='blue', label='conditional mean')
        ax.plot(val_x_axis, self.lower_bound, 'g--', label='95 aleatoric interval')
        ax.plot(val_x_axis, self.upper_bound, 'g--')
        ax.legend()

        ax.set_ylabel('Revenue')        
        ax.set_xlabel('validation')
        ax.set_title('Aleatoric uncertainy')
        # ax.set_ylim([0, 6])


def show_dist(model):
    dummy_input = np.array([[0]])
    model_prior = model.layers[0]._prior(dummy_input)
    model_posterior = model.layers[0]._posterior(dummy_input)
    print('prior mean:', model_prior.mean().numpy())
    print('prior variacnce', model_prior.variance().numpy())
    print('posterior mean:', model_posterior.mean().numpy())
    # print('posterior covariance', model_posterior.covariance().numpy()[0])
    # print('', model_posterior.covariance().numpy()[1])


def pl_preds_uncertainty(x_axis, predictions, ground_truth, option):
    """ plot uncertainty of preds on a data point """

    fig, ax = plt.subplots()

    if option == 'meanNstd':

        mean = np.mean(predictions, axis=0)
        # median = np.median(predictions, axis=0)
        std = np.std(predictions, axis=0)

        # the ensemble average curve
        ax.plot(x_axis, mean, 'r:', label='ensemble average')
        # ax.plot(x_axis, median, 'r:', label='ensemble median')
        ax.fill_between(x_axis,
                        mean + 2 * std,
                        mean - 2 * std,
                        color='salmon',
                        alpha=0.3,
                        label='mean +- 2 sigma')
        # 3 * sigma
        ax.fill_between(x_axis,
                        mean + 3 * std,
                        mean - 3 * std,
                        color='salmon',
                        alpha=0.15,
                        label='median +- 3 sigma')
    elif option == 'CI95':
        CI95_metric = CI95_interval(predictions)

        # the ensemble median
        ax.plot(x_axis, CI95_metric.PI_median,
                color='red', label='ensemble median')
        ax.fill_between(x_axis,
                        CI95_metric.PI_2p5,
                        CI95_metric.PI_97p5,
                        color='coral',
                        alpha=0.2,
                        label=r'95\%' ' credible interval')
    elif option == 'multi':
        for c, row in enumerate(predictions):
            if c == 0:
                ax.plot(x_axis, row, 'gray', alpha=0.1,
                        zorder=0, label='prediction')
            else:
                ax.plot(x_axis, row, 'gray', alpha=0.1, zorder=0)

    # plot ground truth
    ax.scatter(x_axis, ground_truth, color='b', label='target', zorder=1)

    ax.set_title('Epistemic uncertainty')
    ax.set_xlabel('Validation')
    ax.set_ylabel('Revenue')
    # ax.set_ylim([0, 10])

    ax.legend(frameon=False, loc='upper left', fontsize='small')
    ax.grid(ls=':')


def plot_history_info(history_obj, title=''):
    """ given a tf history object, display the learning curve of loss and metric """

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

    a, b, c, d, e, f = history_obj.history.keys()

    ax[0].plot(history_obj.epoch, history_obj.history[a], label=a)
    ax[0].plot(history_obj.epoch, history_obj.history[d], label=d)
    ax[0].set_ylabel(a)
    ax[0].set_xlabel('epochs')
    ax[0].legend()

    ax[1].plot(history_obj.epoch, history_obj.history[b], label=b)
    ax[1].plot(history_obj.epoch, history_obj.history[e], label=e)
    ax[1].set_ylabel(b)
    ax[1].set_xlabel('epochs')
    ax[1].legend()

    ax[2].plot(history_obj.epoch, history_obj.history[c], label=c)
    ax[2].plot(history_obj.epoch, history_obj.history[f], label=f)
    ax[2].set_ylabel(c)
    ax[2].set_xlabel('epochs')
    ax[2].legend()

    fig.suptitle(title)


def CI95_interval(ensemble_predictions):
    """ compute the 95 credible interval 

    Parameters
    ----------
    ensemble_predictions : array
        2D, e.g. (100*8),  matrix of ensemble predictions of the test set
    """

    PI_2p5 = np.percentile(a=ensemble_predictions, q=2.5, axis=0)
    PI_97p5 = np.percentile(a=ensemble_predictions, q=97.5, axis=0)
    PI_median = np.median(ensemble_predictions, axis=0)

    return CI95_metrics(PI_2p5, PI_97p5, PI_median)


def cp_ensemble_dp(model, input_dp, ensemble_size):
    """ Possibly a even more low-level  computation of an ensemble of means and stds (for all frequencies), respectively,
    for epistemic uncertainty and aleatory uncertainty.

    ! the computation part of the above function `pl_mixed_uncertainty_PI`

    Parameters
    ----------
    input_dp : array in shape (33,)
        the input data point from, say, the test set (in np.arrar)
    groundTruth: array
        the corresponding ground truth from the test set
    option : str,
        Two options, either output the plot only, or return the uncertainty metrics, PI width and PICP

    Ruturn
    ------
    An ensemble of `mean` and `std` of tfp.distribution objects for each data point;
    """

    dist_objs = [model(input_dp) for _ in range(ensemble_size)]
    means = [np.squeeze(dist_obj.mean()) for dist_obj in dist_objs]
    stds = [np.squeeze(dist_obj.stddev()) for dist_obj in dist_objs]

    # each dict element is a `ensembles_size` length of list of arrays, each array in (, 33)
    return {'ensemble_means': means, 'ensemble_stds': stds}


def cp_dp_PI_bound(ensemble_dp_dist, style='envelop', k=2):
    """ the **envelop** style method for computing the PI bounds for a data point 
    ! the computation part of the above function `pl_mixed_uncertainty_PI`

    Parameters
    ----------
    input_dp : array in shape (33,)
        the input data point from, say, the test set (in np.arrar)
    groundTruth: array
        the corresponding ground truth from the test set
    k : int 
        confidence level, normally 2 or 3
    style : str,
        Two style of computing the PI bounds from the same ensemble distribution results.

    Ruturn
    ------
    the uncertainty width metric for a certain data point
    """

    means = ensemble_dp_dist['ensemble_means']
    stds = ensemble_dp_dist['ensemble_stds']

    if style == 'envelop':
        # one way to yield the  PI ([lb, ub]) - envelop style
        lb = [mean - k * std for mean, std in zip(means, stds)]
        ub = [mean + k * std for mean, std in zip(means, stds)]

        lower_bound_curve = np.amin(np.vstack(lb), axis=0)
        upper_bound_curve = np.amax(np.vstack(ub), axis=0)
        mean_curve = np.mean(np.vstack(means), axis=0)

    elif style == 'gmm':
        ensemble_means = np.vstack(means)
        ensemble_stds = np.vstack(stds)
        gmm_lists_dp = []
        # of each frequency, cretea a uniform mixure-of-Gaussians
        for col in range(ensemble_means.shape[1]):
            gm_f = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    probs=[1 / ensemble_means.shape[0]] * ensemble_means.shape[0]),
                components_distribution=tfd.Normal(
                    loc=ensemble_means[:, col],
                    scale=ensemble_stds[:, col],
                )
            )
            gmm_lists_dp.append(gm_f)

        mean_curve = np.array([gmm.mean().numpy() for gmm in gmm_lists_dp])
        std_gmm = np.array([gmm.stddev().numpy() for gmm in gmm_lists_dp])
        lower_bound_curve = mean_curve - 2 * std_gmm
        upper_bound_curve = mean_curve + 2 * std_gmm

    return PI_bound(ensembleAverage=mean_curve, lowerBound=lower_bound_curve, upperBound=upper_bound_curve, mean=mean_curve, std=std_gmm)


def pl_priorNposterior(trace, parameter, prior_obj, low=-2, high=2):
    """ plot both the prior and posterior distribution of a parameter 
    
    parameters
    ----------
    trace : pymc3 trace object
        the trace from which to get posterior samples
    parameter : str
        which parameter to plot
    prior_obj : tfp distribution object
        Tensorflow probability distribution object as prior with given parameters and type
    low : float
        lower bound of the parameter on the x-axis
    high : float
        upper bound of the parameter on the x-axis
    """

    # get the posterior of a parameter (i.e. samples in np array)
    pos_dist = trace.posterior[parameter].to_numpy()[0]

    # manually create and plot the prior distribution

    # the posterior 
    fig, ax = plt.subplots()
    dummy_xaxis = np.linspace(low, high, 100)
    ax.plot(dummy_xaxis, prior_obj.prob(dummy_xaxis), color='green', label='prior')
    sns.histplot(x=pos_dist, bins=10, kde=True, stat='density', label='posterior', ax=ax)
    ax.legend()
    ax.set_xlabel(f'{parameter}')
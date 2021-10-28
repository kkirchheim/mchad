import logging
import numpy as np
log = logging.getLogger(__name__)

try:
    import libmr
except ImportError as e:
    log.error("You have to insall 'libmr' and 'cython' to use OpenMax")
    raise e


class OpenMax(object):
    """
    Implementation of the OpenMax Layer as proposed by Bendale et. al in *Towards Open Set Deep Networks*.

    :param tailsize: length of the tail to fit the distribution to
    :param alpha: number of class activations to revise
    :param euclid_weight: weight for the euclidean distance.

    :see Paper: https://arxiv.org/abs/1511.06233
    :see Implementation: https://github.com/abhijitbendale/OSDN
    """

    def __init__(self, tailsize=25, alpha=10, euclid_weight=1.0):
        self.tailsize = tailsize
        self.alpha = alpha

        self.euclid_weight = euclid_weight
        self.cos_weight = 1 - euclid_weight

        self.centers = None  # class centers, i.e. MAVs
        self.n_dims = None
        self.distributions = None  # weibull distributions
        self.is_fitted = None

        self._reset()

    def _reset(self):
        self.centers = dict()
        self.n_dims = None
        self.distributions = dict()
        self.is_fitted = False

    def fit(self, x, y):
        """
        Fit Openmax layer

        :param x: class activations/logits for samples
        :param y: labels for samples
        """
        self._reset()
        classes = np.unique(y)
        self.n_dims = x.shape[1]

        assert x is not None
        assert y is not None
        assert self.alpha is not None

        log.debug(f"Input shape: {x.shape}")
        self.alpha = min(x.shape[1], self.alpha)

        for n, clazz in enumerate(classes):
            idxs = y == clazz

            if self.centers.get(clazz) is None:
                self.centers[clazz] = self._get_center(x[idxs])
            else:
                pass

            # calculate distances of all elements
            dists = self._get_dists_to_center(clazz, x[idxs])

            tailtofit = sorted(dists)[-self.tailsize:]

            model = libmr.MR(alpha=self.alpha)
            model.fit_high(tailtofit, len(tailtofit))

            if not model.is_valid:
                log.error(f"Fitting was invalid for class {clazz}: {len(tailtofit)} instances)")

            self.distributions[clazz] = model

        self.is_fitted = True
        return self

    def predict(self, x) -> np.ndarray:
        """
        Calculate revised activation vector.

        :param x: class activations/logits for samples

        :returns: revised activation vector
        """
        assert self.n_dims == x.shape[1]

        log.debug(f"OpenMax: predicting for input shape {x.shape}")
        labels = np.argmax(x, axis=1)
        classes = np.unique(labels)

        # distance of each point to each cluster center, this vectorized operation is several orders of magn. faster
        # then individual queries
        distances = np.zeros((x.shape[0], self.n_dims))

        for clazz in classes:
            try:
                log.debug(f"Calculating distances for class {clazz}")
                distances[:, clazz] = self._get_dists_to_center(clazz, x)
            except KeyError:
                distances[:, clazz] = np.full((distances.shape[0],), 1e12)  # TODO: set to high value

        # buffer for results
        revised_activation = np.zeros((x.shape[0], x.shape[1] + 1))

        # get indexes of top predictions, in ascending order
        top_predictions = np.argsort(x, axis=1)[:, -self.alpha::][:, ::-1]
        top_predictions = top_predictions[:, :self.alpha]

        # alpha weights should be the same for each prediction
        alpha_weights = [((self.alpha + 1) - i) / float(self.alpha) for i in range(1, self.alpha + 1)]

        # loop through all predictions (a-1)/a for elements in [1, alpha]
        for j, (activation, top_prediction) in enumerate(zip(x, top_predictions)):


            # calculate ranked alpha
            # ranked_alpha will be zero for all but the top predictions
            ranked_alpha = np.zeros((x.shape[1],))
            ranked_alpha[top_prediction] = alpha_weights

            # for ignored classes (not in top classes), this value will be 1 (as initialized)
            # for classes in close range to the cluster center, this value should be high (?)
            ws = np.ones((x.shape[1]))

            for i, pred_class in enumerate(top_prediction):
                # calculate distance
                dist = distances[j, pred_class]

                # get probability of instance being an outlier.
                # if high -> outlier
                # low -> no outlier
                try:
                    w = self.distributions[pred_class].w_score(dist)
                    ws[pred_class] = w
                except KeyError:
                    # print(f"Error: class not found: {pred_class}")
                    pass

            wscores = 1 - ws * ranked_alpha  # wscores will be 1 except for the top predictions

            # now that we have calculated the weights, calc the revised activation vector
            revised_activation[j, 1:] = activation * wscores

            # calculate proxy activation for outlier class
            # will be high if wscores is low
            v_0 = np.sum(activation * (1 - wscores))
            revised_activation[j, 0] = v_0

        # scale with softmax
        # NOTE: every integer value x > 709 will lead to an overflow in e^x
        # See https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function#40726641
        if np.max(revised_activation) > 709:
            log.warning("Rounding because of high revised activation vector")
        revised_activation = np.minimum(revised_activation, 709)
        e = np.exp(revised_activation)
        p = e / np.sum(e, axis=1)[:, None]

        return p

    def _get_center(self, x):
        return np.mean(x, axis=0)

    def _cos_dist(self, x, y):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return np.dot(x, y) / (norm_x * norm_y)

    def _get_dists_to_center(self, clazz, x):
        center = self.centers[clazz]

        if self.cos_weight == 0:
            euclid_dist = np.linalg.norm(x - center, axis=1)
            return euclid_dist

        if self.euclid_weight == 0:
            cos_dist = self._cos_dist(center, x)
            return cos_dist

        euclid_dist = np.linalg.norm(x - center, axis=1)
        cos_dist = self._cos_dist(center, x)

        # calculate weighted distance
        d = self.cos_weight * cos_dist + self.euclid_weight * euclid_dist
        return d

    def _cos_dist(self, center, x):
        cos_dist = np.ndarray((x.shape[0],))
        norm_x = np.linalg.norm(x, axis=1)
        norm_center = np.linalg.norm(center)
        for i in range(x.shape[0]):
            cos_dist[i] = 1 - np.dot(x[i], center) / (norm_x[i] * norm_center)
        return cos_dist

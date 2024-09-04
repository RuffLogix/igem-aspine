import os
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

import xpresso_module.constants as constants


class XPressoModel:
    def __init__(self, promoter_shape: Tuple[int], halflife_shape: Tuple[int]) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        promoter_shape : Tuple[int]
            The shape of the promoter data.
        halflife_shape : Tuple[int]
            The shape of the halflife data.

        Examples
        --------
        >>> model = XPressoModel(promoter_shape=(1000, 4, ), halflife_shape=(8,))
        """

        self.promoter_shape = promoter_shape
        self.halflife_shape = halflife_shape
        self._model = self._create_model()

    def __call__(
        self, promoter: npt.NDArray[np.float32], halflife: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Make a prediction using the Xpresso model.

        Parameters
        ----------
        promoter : npt.NDArray[np.float32]
            The promoter data.
        halflife : npt.NDArray[np.float32]
            The halflife data.

        Returns
        -------
        npt.NDArray[np.float32]
            The model's prediction.

        Examples
        --------
        >>> model = XPressoModel(promoter_shape=(1000, 4, ), halflife_shape=(8,))
        >>> promoter = np.random.rand(1, 1000, 4).astype(np.float32)
        >>> halflife = np.random.rand(1, 8).astype(np.float32)
        >>> model(promoter, halflife)
        array([0.5], dtype=float32)
        """

        return self._model.predict([promoter, halflife], batch_size=64).flatten()

    def fit(
        self,
        train_promoter: npt.NDArray[np.float32],
        train_halflife: npt.NDArray[np.float32],
        train_y: npt.NDArray[np.float32],
        valid_promoter: npt.NDArray[np.float32],
        valid_halflife: npt.NDArray[np.float32],
        valid_y: npt.NDArray[np.float32],
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        n_epochs: int = constants.DEFAULT_N_EPOCHS,
        save_dir: str = constants.DEFAULT_SAVE_DIR,
    ) -> Dict[str, List[float]]:
        """
        Fit the model to the training data.

        Parameters
        ----------
        train_promoter : npt.NDArray[np.float32]
            The promoter data for the training set.
        train_halflife : npt.NDArray[np.float32]
            The halflife data for the training set.
        train_y : npt.NDArray[np.float32]
            The target data for the training set.
        valid_promoter : npt.NDArray[np.float32]
            The promoter data for the validation set, by default None.
        valid_halflife : npt.NDArray[np.float32]
            The halflife data for the validation set, by default None.
        valid_y : npt.NDArray[np.float32]
            The target data for the validation set, by default None.
        batch_size : int, optional
            The batch size to use during training, by default 64.
        n_epochs : int, optional
            The number of epochs to train for, by default 250.
        save_dir : str, optional
            The directory to save the model's best parameters, by default "./checkpoints".

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the training and validation loss and accuracy history.

        Examples
        --------
        >>> model = XPressoModel(promoter_shape=(1000, 4, ), halflife_shape=(8,))
        >>> promoter = np.random.rand(1, 1000, 4).astype(np.float32)
        >>> halflife = np.random.rand(1, 8).astype(np.float32)
        >>> y = np.random.rand(1).astype(np.float32)
        >>> model.fit(promoter, halflife, y, promoter, halflife, y)
        {'loss': [0.5], 'val_loss': [0.5], 'mean_squared_error': [0.5], 'val_mean_squared_error': [0.5]}
        """

        save_path = os.path.join(save_dir, constants.DEFAULT_CHECKPOINT_NAME)

        # Create a callback that saves the model's weights
        checkpoint_callback = ModelCheckpoint(
            save_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )

        # Create a callback that stops the training when the validation loss stops improving
        earlystopping_callback = EarlyStopping(
            monitor="val_loss", patience=7, verbose=1, mode="min"
        )

        # Fit the model
        result = self._model.fit(
            [train_promoter, train_halflife],
            train_y,
            batch_size=int(batch_size),
            shuffle=True,
            epochs=n_epochs,
            validation_data=([valid_promoter, valid_halflife], valid_y),
            callbacks=[earlystopping_callback, checkpoint_callback],
        )

        return result.history

    def _create_model(self) -> Model:
        """
        Create the Xpresso model.
        """

        activationFxn = "relu"

        halflifedata = Input(shape=self.halflife_shape, name="halflife")
        input_promoter = Input(shape=self.promoter_shape, name="promoter")

        x = Conv1D(
            int(2**7),
            int(6),
            dilation_rate=int(1),
            padding="same",
            kernel_initializer="glorot_normal",
            activation=activationFxn,
        )(input_promoter)
        x = MaxPooling1D(int(30))(x)

        maxPool2 = int(10)
        x = Conv1D(
            int(2**5),
            int(9),
            dilation_rate=int(1),
            padding="same",
            kernel_initializer="glorot_normal",
            activation=activationFxn,
        )(x)
        x = MaxPooling1D(maxPool2)(x)

        x = Flatten()(x)
        x = Concatenate()([x, halflifedata])
        x = Dense(int(2**6))(x)
        x = Activation(activationFxn)(x)
        x = Dropout(0.00099)(x)

        x = Dense(int(2))(x)
        x = Activation(activationFxn)(x)
        x = Dropout(0.01546)(x)

        main_output = Dense(1)(x)
        model = Model(inputs=[input_promoter, halflifedata], outputs=[main_output])

        model.compile(
            optimizer=SGD(learning_rate=5 * 1e-4, momentum=0.9),
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )

        return model

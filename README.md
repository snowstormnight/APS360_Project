# APS360_Project
The rapid advancement of drone technology has created exciting opportunities across various fields, from aerial photography to disaster management. However, a challenge remains: accurately predicting a drone's position in real-time, especially in complex environments. Traditional methods like visual-inertial odometry (VIO) rely heavily on visual data, which can be unreliable in low-light conditions or when obstacles obstruct the view. Moreover, this approach requires high computational power, which is difficult to achieve under edge conditions on a drone. These challenges have inspired our project to seek out innovative solutions for more robust and precise drone localization.

Our project aims to develop a system that leverages deep learning, specifically Long Short-Term Memory (LSTM) models, to predict a drone's position using data from inertial measurement units (IMUs). This approach is both interesting and significant because it allows drones to operate independently of external visual cues, enhancing their reliability and versatility. This capability is especially important in scenarios where traditional navigation aids might fail, such as in dark or cluttered environments. Additionally, this approach decreases the computational power required and necessitates less hardware, making drones more accessible to those interested in such areas.

Deep learning is a powerful tool for this task due to its ability to process large amounts of sequential data and uncover complex patterns. LSTM models, a type of recurrent neural network (RNN), are particularly well-suited for time-series prediction, making them ideal for handling the continuous data streams from IMUs. By learning long-term dependencies in this data, LSTMs can provide more accurate and reliable position predictions.

In summary, our project aims to push the boundaries of drone navigation by harnessing the strengths of deep learning. By overcoming the limitations of existing methods, we hope to create a more robust and accurate system for drone localization, contributing to the broader advancement of autonomous navigation technologies.

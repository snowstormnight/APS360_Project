# APS360_Project
Motivations and Goals

Our project seeks to improve drone navigation by developing an inertial-only navigation system based on Transformer models. Unlike traditional methods like visual-inertial odometry (VIO), which rely on visual data and can fail in low-light or obstructed environments, our system uses only inertial measurement units (IMUs). This approach allows drones to navigate effectively in challenging conditions without the need for external visual inputs, reducing the dependence on computationally intensive processes.

Importance and Interest

This project is crucial because it overcomes the limitations of current navigation methods, making drone navigation more robust and adaptable. In environments where visual-based systems are unreliable, such as in darkness or cluttered areas, our inertial-only system ensures continuous and accurate position tracking. This advancement not only enhances the reliability of drones for applications like disaster response and industrial inspections but also reduces hardware complexity and computational demands.

Why Deep Learning?

Deep learning, especially Transformer models, is well-suited for processing IMU data because it excels at capturing long-term dependencies and intricate patterns in sequential data. IMU data is inherently complex and time-dependent, requiring a model that can effectively handle these characteristics. Transformers leverage self-attention mechanisms, allowing them to focus on the most relevant parts of the sequence, regardless of their distance, which is crucial for accurate time-series prediction. This architecture not only improves the precision of position predictions but also ensures scalability, making it a powerful approach for advancing autonomous navigation in drones. By implementing Transformers, we can significantly enhance the capabilities of drones in various applications, driving innovation in autonomous technologies. 


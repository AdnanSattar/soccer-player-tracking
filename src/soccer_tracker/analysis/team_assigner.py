"""Team assignment based on player jersey colors."""

from sklearn.cluster import KMeans


class TeamAssigner:
    """Assign players to teams based on jersey colors."""

    def __init__(self):
        """Initialize the team assigner."""
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        """Get KMeans clustering model for image colors.
        
        Args:
            image: Image array (numpy array)
            
        Returns:
            Fitted KMeans model
        """
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """Extract dominant color from player's jersey.
        
        Args:
            frame: Video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Dominant color of the player's jersey
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half = image[0:int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to image shape
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """Assign team colors based on player jersey colors.
        
        Args:
            frame: Video frame
            player_detections: Dictionary of player detections
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Divide colors gotten into 2 teams
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """Get the team assignment for a player.
        
        Args:
            frame: Video frame
            player_bbox: Player bounding box
            player_id: Player track ID
            
        Returns:
            Team ID (1 or 2)
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            raise ValueError("Team colors must be assigned first using assign_team_color()")

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id

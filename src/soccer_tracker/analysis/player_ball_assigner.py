"""Assign ball possession to players."""

from ..utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """Assign ball possession to the nearest player."""

    def __init__(self, max_player_ball_distance=100):
        """Initialize the player-ball assigner.
        
        Args:
            max_player_ball_distance: Maximum distance for ball assignment
        """
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players, ball_bbox):
        """Assign ball to the nearest player within threshold.
        
        Args:
            players: Dictionary of player tracks for current frame
            ball_bbox: Ball bounding box coordinates
            
        Returns:
            Player ID if assigned, -1 otherwise
        """
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

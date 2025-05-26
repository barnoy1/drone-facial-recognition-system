"""Frontend package for the drone facial recognition system.

This package implements the Model-View-Presenter pattern:
- Model: Application state and backend interface
- View: UI components and user interaction
- Presenter: Coordinates between Model and View
"""

from .model.drone_model import DroneModel
from .view.drone_view import DroneView
from .presenter.drone_presenter import DronePresenter

__all__ = ['DroneModel', 'DroneView', 'DronePresenter']

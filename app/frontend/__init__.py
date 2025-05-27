"""Frontend package for the drone facial recognition system.

This package implements the Model-View-Presenter pattern:
- Model: Application state and backend interface
- View: UI components and user interaction
- Presenter: Coordinates between Model and View
"""

from app.frontend.callbacks import DroneModel
from app.frontend.app_view import AppView
from app.frontend.presenter import Presenter

__all__ = ['DroneModel', 'AppView', 'Presenter']

import sys
import argparse
from PySide6.QtWidgets import QApplication

from frontend.model.drone_model import DroneModel
from frontend.view.drone_view import DroneView
from frontend.presenter.drone_presenter import DronePresenter
from backend.config.config_manager import ConfigManager

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Drone Facial Recognition System")
    parser.add_argument('--config', type=str, default='app/settings/config_webcam.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        # Initialize ConfigManager singleton
        config_manager = ConfigManager.instance()
        config_manager.initialize(args.config)

        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create MVP components
        model = DroneModel()
        view = DroneView()
        presenter = DronePresenter(model, view)
        
        # Initialize system
        if not presenter.initialize():
            sys.exit(1)
            
        # Show UI
        view.show()
        
        # Start Qt event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
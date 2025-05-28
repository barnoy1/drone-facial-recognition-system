import sys
import argparse
from PySide6.QtWidgets import QApplication

from app.backend import MissionManager
from app.frontend.callbacks import DroneModel
from app.frontend.app_view import AppView
from app.frontend.presenter import Presenter


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Drone Facial Recognition System")
    parser.add_argument('--config', type=str, default=None, required=True,
                      help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./out', required=False,
                      help='Output directory')
    args = parser.parse_args()
    
    try:


        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create MVP components
        view = AppView()
        presenter = Presenter(args, view)
        
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
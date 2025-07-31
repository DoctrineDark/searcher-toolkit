from .Controller import Controller

class Home(Controller):
    def index():
        return dict([('name', 'Searcher Toolkit'), ('version', 'v1.0.0')]), 200
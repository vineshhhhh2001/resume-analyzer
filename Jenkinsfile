
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }

        stage('Verify Python') {
            steps {
                bat '"C:\\Users\\abhij\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" --version'
                bat '"C:\\Users\\abhij\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m pip --version'
            }
        }

        stage('Backend Tests') {
            steps {
                dir('backend') {
                    bat '"C:\\Users\\abhij\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m pip install -r requirements.txt'
                    bat 'if not exist test-results mkdir test-results'
                    bat '"C:\\Users\\abhij\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m pytest tests -v --tb=short --junitxml=test-results/results.xml'
                }
            }
            post {
                always {
                    junit testResults: 'backend/test-results/results.xml',
                          allowEmptyResults: true
                }
            }
        }
    }

    post {
        always {
            echo 'Jenkins pipeline execution completed.'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check the console output.'
        }
    }
}

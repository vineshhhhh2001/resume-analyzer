
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
                bat 'python --version'
                bat 'python -m pip --version'
            }
        }

        stage('Backend Tests') {
            steps {
                dir('backend') {
                    bat 'python -m pip install -r requirements.txt'
                    bat 'python -m pytest tests -v --tb=short --junitxml=test-results/results.xml'
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

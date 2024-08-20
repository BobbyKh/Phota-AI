#!/bin/bash

# Build static site
hugo

# Configure Firebase
firebase use --add

# Deploy to Firebase
firebase deploy --project snapclick

pip install -r requirements.txt

#!/bin/bash
# Quick launcher for the video analysis pipeline


# python analysis_pipeline/pipeline_driver.py --video-dir my_videos  --output-dir results


set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Video Analysis Pipeline Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$SCRIPT_DIR"
PIPELINE_DIR="$SCRIPT_DIR/analysis_pipeline"

# Default values
VIDEO_DIR="$WORKSPACE_ROOT/best_media"
OUTPUT_DIR="$WORKSPACE_ROOT/data/output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video-dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_pipeline.sh [options]"
            echo ""
            echo "Options:"
            echo "  --video-dir PATH    Directory containing videos (default: best_media)"
            echo "  --output-dir PATH   Output directory (default: data/output)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Example:"
            echo "  ./run_pipeline.sh --video-dir my_videos --output-dir results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo -e "${YELLOW}⚠️  Warning: Video directory not found: $VIDEO_DIR${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Video Directory: $VIDEO_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Pipeline: $PIPELINE_DIR/pipeline_driver.py"
echo ""

# Find Python executable
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${YELLOW}❌ Python not found. Please install Python 3.${NC}"
    exit 1
fi

echo -e "${GREEN}Using Python: $($PYTHON_CMD --version)${NC}"
echo ""

# Run the pipeline
echo -e "${BLUE}Starting pipeline...${NC}"
echo ""

cd "$WORKSPACE_ROOT"

$PYTHON_CMD "$PIPELINE_DIR/pipeline_driver.py" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  📊 video_classifications.csv - Video classifications"
    echo "  📁 templates/ - Template profiles"
    echo "  📈 template_insights.json - Aggregated insights"
    echo "  📉 niche_comparison.png - Visualizations"
else
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}❌ Pipeline failed${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi

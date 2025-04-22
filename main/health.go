package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.

import(
	"fmt"
	"encoding/json"
	"net/http"
)

// health handles the `/health` endpoint to report the current server status.
//
// It returns a JSON-encoded HealthResponse that includes:
//   - `status`: a string representation of the server's internal status
//   - `progress`: any ongoing model loading or initialization progress
//
// This endpoint is typically used for:
//   - Load balancer health checks
//   - Monitoring system probes (e.g., readiness/liveness checks)
//   - Clients polling for model readiness before sending requests
//
// Example response:
// {
//   "status": "ready",
//   "progress": "model loaded successfully"
// }
//
// Response codes:
//   - 200 OK: Health status returned successfully
//   - 500 Internal Server Error: Failed to encode response
func (s *Server) health(w http.ResponseWriter, r *http.Request) {

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&HealthResponse{
		Status:   s.status.ToString(),
		Progress: s.progress,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}
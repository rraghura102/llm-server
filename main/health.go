package main

/**
 *
 * MIT License
 *
 * Copyright (c) 2025 Rayan Raghuram
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
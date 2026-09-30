# Stage 1: Build the React/Vite application
FROM node:20-alpine AS build

WORKDIR /app

# Copy package definitions first for layer caching
COPY package*.json ./

# Install dependencies cleanly
RUN npm install

# Copy application source code
COPY . .

# Build the production assets to /app/dist
RUN npm run build

# Stage 2: Production web server
FROM nginx:alpine

# Remove default nginx static files
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx reverse-proxy and SPA configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose standard web port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

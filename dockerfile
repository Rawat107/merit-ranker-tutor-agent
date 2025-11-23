# -------- Stage 1: Build --------
FROM node:25-alpine3.21 AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# -------- Stage 2: Production --------
FROM node:25-alpine3.21

WORKDIR /app

# Copy built files and node_modules from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/src/public ./src/public

EXPOSE 3000

# Add signal handling utility
RUN apk update && apk upgrade && apk add --no-cache dumb-init

USER node

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]

import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";

export async function GET(
    request: NextRequest,
    { params }: { params: Promise<{ videoId: string }> },
) {
    try {
        const session = await auth();
        const userId = session?.user?.id || "default-user";

        const { videoId } = await params;

        const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
        const backendResponse = await fetch(
            `${backendUrl}/api/analyze/${videoId}/status`,
            {
                method: "GET",
                headers: {
                    "user-id": userId,
                },
            },
        );

        if (!backendResponse.ok) {
            const errorData = await backendResponse.json();
            return NextResponse.json(
                { error: errorData.detail || "Failed to fetch analysis status" },
                { status: backendResponse.status },
            );
        }

        const status = await backendResponse.json();

        return NextResponse.json(status);
    } catch (error) {
        console.error("Fetch analysis status error:", error);
        return NextResponse.json(
            { error: "Failed to fetch analysis status" },
            { status: 500 },
        );
    }
}

import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";

export async function POST(
    request: NextRequest,
    { params }: { params: Promise<{ videoId: string }> },
) {
    try {
        const session = await auth();
        const userId = session?.user?.id || "default-user";

        const { videoId } = await params;

        const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
        const backendResponse = await fetch(
            `${backendUrl}/api/analyze/${videoId}`,
            {
                method: "POST",
                headers: {
                    "user-id": userId,
                },
            },
        );

        if (!backendResponse.ok) {
            const errorData = await backendResponse.json();
            return NextResponse.json(
                { error: errorData.detail || "Failed to trigger analysis" },
                { status: backendResponse.status },
            );
        }

        const result = await backendResponse.json();

        return NextResponse.json(result);
    } catch (error) {
        console.error("Trigger analysis error:", error);
        return NextResponse.json(
            { error: "Failed to trigger analysis" },
            { status: 500 },
        );
    }
}
